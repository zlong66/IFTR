###################################################
## replace rbf_kernel_gram by rbf_kernel_gram_mm ##
###################################################


nuis_cf <- function(data_list, params, cf = 5, nuisance = "prox-pmmr", Kc=NULL){
  set.seed(123)
  n <- nrow(data_list$treatment)
  KA <- data_list$KA
  cf_idx <- sample(rep(1:cf, length.out = n))
  a <- list()
  Kc_testtrain <- list()
  A.train <- list()
  Z.test <- list()
  for (i in 1:cf){
    if(cf==1){trainidx = 1:n}else{trainidx = which(cf_idx != i)}
    testidx <- which(cf_idx == i)
    A.train[[i]] <- data_list$treatment[trainidx,]
    Z.test[[i]] <- data_list$Z[testidx,]
    
    if (nuisance == "prox-pmmr"){
      a[[i]] <- pmmr_approx(Kl = (data_list$K_wx*KA)[trainidx, trainidx], 
                            Kw = (data_list$K_zx*KA)[trainidx, trainidx],
                            Y = data_list$Y[trainidx], 
                            bl = params$pmmr_bl, nystr = params$nystr, 
                            M = params$M, device = params$device)$a
      Kc_testtrain[[i]] <- data_list$K_wx[testidx, trainidx]
    }else if (nuisance == "KRR"){
      a[[i]] <- KRR(KA = (Kc*KA)[trainidx,trainidx], 
                    gA = (Kc*KA)[trainidx,trainidx], 
                    Y = (data_list$Y)[trainidx], 
                    lam = params$select_lam, nystr = params$nystr, 
                    M = params$M, device = params$device)$alpha
      Kc_testtrain[[i]] <- Kc[testidx, trainidx]
    }
  }
  return(list(a = a, cf_idx = cf_idx, 
              Z.test = Z.test, Kc = Kc_testtrain, 
              A.train = A.train))
}

valuecfM <- function(B, a, Z, Theta, A, Kc, sigmaA){
  cf <- length(a)
  values <- lapply(1:cf, function(i) {
    d.test <- Z[[i]]$mm(B)$mm(torch_t(Theta))
    KdA <- rbf_kernel_gram_mm(d.test, A[[i]], sigma = sigmaA)
    result <- Kc[[i]]$mul(KdA)$mm(a[[i]])$mean()
    result
  })
  torch_stack(values)$mean()
}



dL2M <- function(b_init, nuis_list, bases, params, rho, n_iteration=50, lr= 5e-2, device){
  # move the tensors to device
  a <- lapply(nuis_list$a, torch_tensor, device = device)
  A <- lapply(nuis_list$A.train, torch_tensor, device = device)
  Kc <- lapply(nuis_list$Kc, torch_tensor, device = device)
  Z <- lapply(nuis_list$Z.test, torch_tensor, device = device) 
  Theta   <- torch_tensor(bases$Theta,   device = device)
  Sigma.z <- torch_tensor(bases$Sigma.z, device = device)
  R       <- torch_tensor(bases$R,       device = device)

  
  B <- torch_tensor(matrix(b_init, nrow = bases$q, ncol = ncol(Theta), byrow = TRUE),
                    device = device, requires_grad = TRUE)
  sigma <- torch_tensor(params$sigma2, device = device)

  ## train with L-BFGS ##
  opt <- optim_lbfgs(B, lr = lr, line_search_fn = "strong_wolfe")
  
  calc_loss <- function(){
    opt$zero_grad()
    loss <- - valuecfM(B=B, a=a, Z=Z, Theta = Theta, A=A, Kc=Kc, sigmaA=sigma) + rho*
      torch_trace(B$mm(R)$mm(torch_t(B))$mm(Sigma.z))
    loss$backward()
    loss
  }
  
  loss.old <- - valuecfM(B=B, a=a, Z=Z, Theta = Theta, A=A, Kc=Kc, sigmaA=sigma) + rho*
    torch_trace(B$mm(R)$mm(torch_t(B))$mm(Sigma.z))
  for (j in 1:n_iteration){
    opt$step(calc_loss)
    # use updated B to calculate the new value
    v.new <- valuecfM(B=B, a=a, Z=Z, Theta = Theta, A=A, Kc=Kc, sigmaA=sigma) 
    loss.new <- -v.new + rho*torch_trace(B$mm(R)$mm(torch_t(B))$mm(Sigma.z))
    
    if(j %% 10 ==0){
      cat("\nIteration: ", j, "\n")
      cat("Value is: ", as.numeric(v.new), "\n")
    }
    if(abs(as.numeric((loss.new-loss.old)/loss.old)) < 1e-4) break
    loss.old <- loss.new
  }

  return(list(B=B, value = v.new, converge = (j<500)))
}



calc_initial_b <- function(A, Z, Theta, device, ITR_type){
  
  if (ITR_type == "nonlinear"){
    Y <- torch_tensor(A, device = device)
    Z <- torch_tensor(Z, device = device)
    Theta <- torch_tensor(Theta, device = device)
    ZTZ <- torch_t(Z)$mm(Z)
    TTT <- torch_t(Theta)$mm(Theta)
    TYZ <- torch_t(Theta)$mm(torch_t(Y))$mm(Z)
    TYZ_vec <- torch_t(TYZ)$reshape(c(-1,1)) # torch reshape matrix by row
    
    # Kronecker of ZTZ and TTT
    ZTZ_TTT <- torch_kron(ZTZ, TTT)
    ZTZ_TTT_inv <- fast_pinv(ZTZ_TTT)
    b <- ZTZ_TTT_inv$mm(TYZ_vec)
    
  }else if(ITR_type == "linear"){
    y <- as.vector(t(A))
    ZkronT <- kronecker(Z, Theta)
    if(is_singular(t(ZkronT) %*% ZkronT)){
      y <- torch_tensor(y, device = device)
      ZkronT <- torch_tensor(ZkronT, device = device)
      ZTY <- torch_t(ZkronT)$mm(y$unsqueeze(2))
      rm(y)
      ZTZT <- torch_t(ZkronT)$mm(ZkronT)
      ZTZT_inv <- fast_pinv(ZTZT)
      b <- ZTZT_inv$mm(ZTY)
      rm(ZkronT, ZTZT_inv, ZTZT, ZTY)
    }else{
      b <- solve(t(ZkronT) %*% ZkronT) %*% t(ZkronT) %*% y
    }
  }
  return(as.array(b))
}


is_singular <- function(M) {
  tryCatch({
    solve(M)
    FALSE  # If successful, not singular
  }, error = function(e) {
    TRUE   # If error, matrix is singular
  })
}

