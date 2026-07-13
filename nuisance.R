################### estimate the nuisance function using PMMR ###################
JITTER_W <- 1e-12
JITTER_L <- 1e-6
JITTER_LWL <- 1e-8


pmmr_approx <- function(Kl, Kw, Y, bl, nystr = F, M = 3000, device){
  # let lambda = 1/n^2/bl^2
  n <- length(Y)
  W <- Kw + JITTER_W * diag(n)
  L <- bl^2*Kl + JITTER_L * diag(n)

  if (nystr){
    W <- torch_tensor(W, device = device, dtype=torch_float64()) 
    L <- torch_tensor(L, device = device, dtype = torch_float64())
    Y <- torch_tensor(Y, device = device, dtype = torch_float64())$reshape(list(-1,1))
    
    set.seed(199)
    ind <- torch_tensor(sort(sample(1:n, M)), device = device, dtype = torch_int64())
    eig_decomp <- nystroem_decomp_torch(W, ind = ind)
    U <- eig_decomp$U
    V <- eig_decomp$V
    
    W_nystr <- U$mm(V)$mm(torch_t(U))
 
    # --- Regularize eigenvalues of V (two parts: tol + lambda_v) ---
    v <- torch_diag(V)   # length M
    tol       <- 1e-5    # floor for tiny eigenvalues
    lambda_v  <- 1e-5    # ridge added to all eigenvalues
    v_clamped <- torch_clamp(v, min = tol)  # v < tol → tol
    v_eff     <- v_clamped + lambda_v       # final effective eigenvalues
    
    ULU <- torch_t(U)$mm(L)$mm(U)+ torch_diag(1 / v_eff)
    ULU_solve_result <- torch::linalg_solve(ULU/n^2, torch_t(U)$mm(L)/n^2) 
    # equivalent to: X = ULU^{-1} Uᵀ L
    B <- torch_eye(n, device = device, dtype = ULU$dtype) - U$mm(ULU_solve_result)
    # a <- bl^2 * B$mm(W_nystr)$mm(Y)
    a <- bl^2 * B$mm(W)$mm(Y)
    Y.pred <- L$mm(a)/bl^2

  }else{
    LWL <- (L %*% W %*% L + L)/n^2
    LWL_inv <- solve(LWL+JITTER_LWL*diag(n))
    a <- bl^2 * LWL_inv %*% L %*% W %*% Y/n^2
    Y.pred <- L%*%a /bl^2
  }
  return(list(a=as.array(a), Y.pred=as.array(Y.pred)))
}


pmmr_tune <- function(Kl, Kw, Y, bl.interval = c(0.001,0.5), device,
                      method = "cv", cv = 5, nystr = F, M = 300, m = 100){
  n <- length(Y)
  if (method == "cv"){
    # bl.seq <- logseq(bl.interval[1], bl.interval[2], 5)
    bl.seq <- seq(bl.interval[1], bl.interval[2], len = 30)
    set.seed(123)
    fold <- sample(rep(1:cv, length.out = n))
    scores <- matrix(NA, ncol = length(bl.seq), nrow = cv)
    
    for (i in 1:cv){
      trainidx <- (fold != i)
      testidx <- (fold == i)
      Y.test <- Y[testidx]
      n.test <- length(Y.test)
      for (j in 1:length(bl.seq)){
        print(j)
        pmmr_train <- pmmr_approx(Kl=Kl[trainidx, trainidx], Kw=Kw[trainidx,trainidx], Y=Y[trainidx], 
                                  bl = bl.seq[j], nystr = nystr, M = M, device = device)
        Y.pred.test <- Kl[testidx, trainidx] %*% pmmr_train$a
        W.test <- Kw[testidx, testidx]
        # mmr_v: t(Y-f(x))Kw(Y-f(x))/n^2
        scores[i,j] <- t(Y.test - Y.pred.test) %*% W.test %*% (Y.test - Y.pred.test) / n.test^2
      }
    }
    score_mean <- apply(scores, 2, mean)
    best.bl <- bl.seq[which.min(score_mean)]
  }else if(method == "lmo"){
    W <- Kw + JITTER_W * diag(n)
    
    if(nystr){
      W <- torch_tensor(W, device = device, dtype=torch_float64()) 
      Y <- torch_tensor(Y, device = device, dtype=torch_float64())$reshape(list(-1,1))
      set.seed(199)
      ind <- torch_tensor(sort(sample(1:n, M)), device = device, dtype = torch_int64())
      eig_decomp <- nystroem_decomp_torch(W, ind = ind)
      U <- eig_decomp$U
      V <- eig_decomp$V
      W_nystr <- U$mm(V)$mm(torch_t(U))
      
      v <- torch_diag(V)   
      tol       <- 1e-5    # floor for tiny eigenvalues
      lambda_v  <- 1e-5    # ridge added to all eigenvalues
      v_clamped <- torch_clamp(v, min = tol)  # v < tol → tol
      v_eff     <- v_clamped + lambda_v       # final effective eigenvalues
    }
    
    lmoerr <- function(bl){
      L <- bl^2*Kl + JITTER_L * diag(n)
      
      if (nystr){
        L <- torch_tensor(L, device = device, dtype=torch_float64())
        ULU <- torch_t(U)$mm(L)$mm(U)+ torch_diag(1 / v_eff)
        ULU_solve_result <- torch::linalg_solve(ULU/n^2, torch_t(U)$mm(L)/n^2)  
        B <- torch_eye(n, device = device, dtype = ULU$dtype) - U$mm(ULU_solve_result)
        C <- L$mm(B)
        # c <- C$mm(W_nystr)$mm(Y)
        c <- C$mm(W)$mm(Y)
        c_y <- c - Y
        W <- as.array(W)
        C <- as.array(C)
        c <- as.array(c)
        c_y <- as.array(c_y)
      }else{
        LWL <- (L %*% W %*% L + L)/n^2
        LWL_inv <- solve(LWL+JITTER_LWL*diag(n))
        C <- L %*% LWL_inv %*% L /n^2
        c <- C %*% W %*% Y
        c_y <- c - Y
      }
      
      lmo_err <- 0
      N <- 0
      
      for (i in seq(1, n, by = m)){
        idxs <- i:(i+m-1)
        K_i <- W[idxs, idxs]
        C_i <- C[idxs, idxs]
        c_y_i <- c_y[idxs]
        b_y <- solve(diag(m) - C_i %*% K_i) %*% c_y_i # mistake 1 (05/16)
        lmo_i <- t(b_y) %*% K_i %*% b_y
        lmo_err <- lmo_err + lmo_i
        N <- N+1
      }
      cat('LMO-err: ', lmo_err[1, 1] / N / m^2, "\n")
      return(lmo_err[1, 1] / N / m^2)
    }
    # find bl that minimize lmoerr
    res <- optimize(lmoerr, interval = bl.interval) # bl.interval = c(0.001, 0.5)
    best.bl <- res$minimum    
    score_mean <- res$objective   # mistake 2 (05/16)
    cat("The best bl is: ", best.bl, "\n")
  }
  return(list(best.bl=best.bl, score_mean = score_mean))
}
  

################### estimate the CATE using KRR ########################
### estimate m using KRR; find the optimal lambda
KRR <- function(KA, gA, Y, lam, nystr=T, M=300, device) {
  n <- nrow(KA)
  
  if(nystr){
    
    KA <- torch_tensor(KA, device = device, dtype = torch_float64())
    Y <- torch_tensor(Y, device = device, dtype = torch_float64())$reshape(list(-1,1))
    EYEN <- torch_eye(n, device = device, dtype = torch_float64())
    
    # ind <- sample(1:n, size = M)
    set.seed(123)
    ind <- torch_tensor(sort(sample(1:n, M)), device = device, dtype = torch_int64())
    eig_decomp <- nystroem_decomp_torch(KA, ind = ind)
    U <- eig_decomp$U
    V <- eig_decomp$V
    
    v <- torch_diag(V)   # length M
    tol       <- 1e-5    # floor for tiny eigenvalues
    lambda_v  <- 1e-5    # ridge added to all eigenvalues
    v_clamped <- torch_clamp(v, min = tol)  # v < tol → tol
    v_eff     <- v_clamped + lambda_v       # final effective eigenvalues
    
    UtU <- n*lam*torch_diag(1/v_eff) + torch_t(U)$mm(U)
    UtU_solve_result <- torch::linalg_solve(UtU, torch_t(U)) 
    C <- 1/(n*lam) * (EYEN - U$mm(UtU_solve_result))
    
    alpha <- C$mm(Y)
    Yhat <- torch_tensor(gA, device=device, dtype = torch_float64())$mm(alpha)

  }else{
    alpha <- solve(KA + n* lam * diag(n), Y)
    alpha <- matrix(alpha, ncol=1)
    Yhat <- gA %*% alpha
  }
  return(list(alpha=as.array(alpha), Yhat=as.array(Yhat)))
}


KRR_tune <- function(KA, Y, lam_seq, criterion = "cv", device,
                     folds = 5, alpha = 1, nystr = T, M = 300) {
  n <- nrow(KA)
  I <- diag(n)
  valerrors <- rep(0, length(lam_seq))
  for (i in 1:length(lam_seq)) {
    print(i)
    if (criterion == "gcv" || criterion == "loo") {
      if (nystr){
        lam <- lam_seq[i]
        KA <- torch_tensor(KA, device = device, dtype = torch_float64())
        Y <- torch_tensor(Y, device = device, dtype = torch_float64())$reshape(list(-1,1))
        EYEN <- torch_eye(n, device = device, dtype = torch_float64())
        # KA_inv <- linalg_inv(KA + n * lam_seq[i] * EYEN)
        set.seed(123)
        ind <- torch_tensor(sort(sample(1:n, M)), device = device, dtype = torch_int64())
        eig_decomp <- nystroem_decomp_torch(KA, ind = ind)
        U <- eig_decomp$U
        V <- eig_decomp$V

        v <- torch_diag(V)   # length M
        tol       <- 1e-5    # floor for tiny eigenvalues
        lambda_v  <- 1e-5    # ridge added to all eigenvalues
        v_clamped <- torch_clamp(v, min = tol)  # v < tol → tol
        v_eff     <- v_clamped + lambda_v       # final effective eigenvalues
        
        UtU <- n*lam*torch_diag(1/v_eff) + torch_t(U)$mm(U)
        UtU_solve_result <- torch::linalg_solve(UtU, torch_t(U)) 
        KA_inv_appx <- 1/(n*lam) * (EYEN - U$mm(UtU_solve_result))
        
        A_mu <- KA$mm(KA_inv_appx)
      }else{
        A_mu <- KA %*% solve(KA + n * lam_seq[i] * I)
      }
      
    }
    if (criterion == "gcv") {
      if (nystr){
        numer <- (EYEN - A_mu)$mm(Y)
        denom <- torch_sum(torch_diag(EYEN - A_mu)/n)
        valerrors[i] <- as.array(torch_mean((numer/denom)^2))
      }else{
        inside <- sum(diag((I - alpha * A_mu) / n))
        if (inside < 0) {
          denor <- 0
        } else {
          denor <- (inside)^2
        }
        numer <- mean(((I - A_mu) %*% Y)^2)
        valerrors[i] <- numer / denor
      }
    }
    if (criterion == "loo") {
      
      if(nystr){
        num <- (EYEN - A_mu)$mm(Y)
        denom <- torch_diag(EYEN - A_mu)
        valerrors[i] <- as.array(torch_mean((num/denom)^2))
      }else{
        valerrors[i] <- mean(((I - A_mu) %*% Y / diag(I - A_mu))^2)
      }
      
    }
    if (criterion == "cv") {
      set.seed(123)
      fold_idx <- sample(rep(1:folds, length.out = n))
      for (fold in 1:folds) {
        trainidx <- (fold_idx != fold)
        validx <- (fold_idx == fold)
        Yeval <- KRR(KA[trainidx, trainidx], KA[validx, trainidx], Y[trainidx], 
                     lam = lam_seq[i], nystr = nystr, M = M, device = device)$Yhat
        valerrors[i] <- valerrors[i] + mean((Y[validx] - Yeval)^2)
      }
    }
  }
  select_lam <- lam_seq[which.min(valerrors)]
  return(list(select_lam = select_lam, valerrors = valerrors))
}

