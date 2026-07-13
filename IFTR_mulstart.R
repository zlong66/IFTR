
tune_pars <- function(data_list, Kc, params, nuisance, nuis_vars, tune_args){
  
  crit <- tune_args$crit
  folds <- tune_args$folds
  lam_seq <- tune_args$lam_seq
  bl_interval <- tune_args$bl_interval
  m <- tune_args$m
  
  if(nuisance == "prox-pmmr"){
    params$pmmr_bl <- pmmr_tune(Kl = data_list$K_wx*data_list$KA, 
                                Kw = data_list$K_zx*data_list$KA, 
                                Y = data_list$Y,
                                method = crit,
                                bl.interval = bl_interval, 
                                m = m,
                                device = params$device,
                                nystr = params$nystr, 
                                M = params$M)$best.bl
  }else if(nuisance == "KRR"){
    params$select_lam <- KRR_tune(KA = Kc*data_list$KA, 
                                  Y=data_list$Y, 
                                  lam_seq=lam_seq, 
                                  criterion = crit, 
                                  folds = folds,
                                  nystr = params$nystr, 
                                  M = params$M, 
                                  device = params$device)$select_lam
    cat("The best lambda is ", params$select_lam)
    
    # calculate the krr_lam_orc
    if(nuis_vars == "UX"){ params$krr_lam_orc <- params$select_lam }
    if(is.null(params$krr_lam_orc)){
      params$krr_lam_orc <- KRR_tune(KA = data_list$K_ux*data_list$KA,
                                     Y=data_list$Y,
                                     lam_seq=lam_seq, 
                                     criterion = crit, 
                                     folds = folds, 
                                     nystr = params$nystr, 
                                     M = params$M, 
                                     device = params$device)$select_lam
      cat("The best lambda for oracle method is ", params$krr_lam_orc)
    }
  }
  return(params)
}


fitIFTRL2 <- function(train, nuisance, nuis_vars, d_covars, params, tune_args,
                      K=5, cf=5, n_iteration=500, lr= 5e-2){
  
  device <- params$device
  params$sigma2 <- get_bw_A(torch_tensor(train$treatment))
  t_seq <- seq(0,1, length = ncol(train$treatment)) 
  rhos <- params$rhos
  
  # standardize the train data
  preProcValues <- preProcess(train$C, method = c("center", "scale"))
  train$C_std <- predict(preProcValues, train$C)
  
  # calculate the Gram matrices and bases
  train_list <- prepare_data(data = train, nuisance = nuisance, params = params)
  bases <- gen_basisL2(params,data=train_list$C_std[,column_list[[d_covars]]], 
                       type=params$ITR_type, k.beta = params$k.beta, t_seq = t_seq)
  train_list <- c(train_list, bases)

  # tune the parameters 
  Kc <- switch(nuis_vars,
               "WXZ" = train_list$K_wxz,
               "WX" = train_list$K_wx,
               "UX" = train_list$K_ux,
               "X" = train_list$K_x)
  params <- tune_pars(train_list, Kc, params, nuisance, nuis_vars, tune_args)
  
  # calculate nuisance functions used for cv
  nuis_cv <- nuis_cf(train_list, params, cf = K, nuisance = nuisance, Kc=Kc)
  cv_idx <- nuis_cv$cf_idx
  
  # cross-validation
  valueCV <- matrix(NA, nrow = length(rhos), ncol = K)

  sigmaA <- as.numeric(params$sigma2)
  Theta   <- torch_tensor(bases$Theta,  device = device, dtype = torch_float64())
  # --- Define penalty matrices (dimensions must match B):
  Sigma.z <- torch_tensor(bases$Sigma.z, device = device, dtype = torch_float64())
  R       <- torch_tensor(bases$R,       device = device, dtype = torch_float64())

  # --- B shape:
  d1 <- bases$q
  d2 <- as.integer(Theta$size(2))
  
  # compute the initial beta values
  if(calc_b0) {
    b_init <- calc_initial_b(A=train_list$treatment, Z=train_list$Z, Theta=train_list$Theta,
                             device = device, ITR_type = params$ITR_type)
    B_warm <- matrix(b_init, nrow = d1, ncol = d2, byrow = TRUE)
    B_warm_torch <- torch_tensor(B_warm, device=device, dtype=torch_float64())
  }else{
    B_warm_torch <- NULL
    }
    
  for (i in 1:K){
    cat("\nCV ", i, " start.")
    # use data excluding fold i to fit nuisance functions
    trainidx <- which(cv_idx != i)
    cv_train_i <- list(treatment = train_list$treatment[trainidx,],
                       Z = train_list$Z[trainidx,],
                       K_wx = train_list$K_wx[trainidx,trainidx],
                       K_zx = train_list$K_zx[trainidx,trainidx],
                       KA = train_list$KA[trainidx,trainidx],
                       Y = train_list$Y[trainidx])
    # use train data to calculate nuisance functions
    nuis_list_i <- nuis_cf(cv_train_i, params, cf = cf, nuisance = nuisance, Kc=Kc[trainidx, trainidx])
    rm(cv_train_i)
    
    # define torch objects
    a_torch <- lapply(nuis_list_i$a, torch_tensor, device = device, dtype = torch_float64())
    Z_torch <- lapply(nuis_list_i$Z.test, torch_tensor, device = device, dtype = torch_float64()) 
    A_torch <- lapply(nuis_list_i$A.train, torch_tensor, device = device, dtype = torch_float64())
    Kc_torch <- lapply(nuis_list_i$Kc, torch_tensor, device = device, dtype = torch_float64())
    
    # obtain the IFTRs for each rho
    for(j in seq_along(rhos)){
 
      # --- Penalty weight:
      rho <- rhos[j]
      
      # run multi-start
      res <- multi_start_opt(
        n_starts = n_starts,
        B_warm = B_warm_torch,
        B_glasso = NULL,
        init = init,
        init_scale = 0.02,
        seed = 123,
        d1 = d1, d2 = d2,
        a = a_torch, Z = Z_torch, Theta = Theta, A = A_torch, Kc = Kc_torch, sigmaA = sigmaA,
        rho = rho, R = R, Sigma.z = Sigma.z,
        adam_lr = adam_lr,
        adam_steps = adam_steps,
        clip_norm = 1.0,
        lbfgs_lr = lbfgs_lr,
        lbfgs_history_size = 10L,
        lbfgs_max_iter = lbfgs_max_iter,
        lbfgs_tol_abs = 1e-6,
        lbfgs_tol_rel = 1e-4,
        verbose = TRUE
      )

      best_B <- res$best_B
      best_obj <- res$best_obj
      cat("best objective is ", best_obj, "\n")
      
      d.test.i <- torch_tensor(nuis_cv$Z.test[[i]], device = device, dtype = torch_float64())$
        mm(best_B)$mm(Theta$t())
      
      KdA.i <- rbf_kernel_gram_mm(
        d.test.i,
        torch_tensor(nuis_cv$A.train[[i]], device = device, dtype = torch_float64()),
        sigma = sigmaA
      )
      
      valueCV[j, i] <- as.numeric(
        torch_tensor(nuis_cv$Kc[[i]], device = device, dtype = torch_float64())$
          mul(KdA.i)$
          mm(torch_tensor(nuis_cv$a[[i]], device = device, dtype = torch_float64()))$
          mean()$item()
      )
    }
  }
  
  # find the best rho
  Vmean <- apply(valueCV, 1, mean)
  max_index <- max(which(Vmean == max(Vmean))) # in case there are multiple max values
  best_rho <- rhos[max_index]
  cat("The best rho is ", best_rho)
  
  # get the FITR using the best rho
  a_torch <- lapply(nuis_cv$a, torch_tensor, device = device, dtype = torch_float64())
  Z_torch <- lapply(nuis_cv$Z.test, torch_tensor, device = device, dtype = torch_float64()) 
  A_torch <- lapply(nuis_cv$A.train, torch_tensor, device = device, dtype = torch_float64())
  Kc_torch <- lapply(nuis_cv$Kc, torch_tensor, device = device, dtype = torch_float64())
  
  res <- multi_start_opt(
    n_starts = n_starts,
    B_warm = B_warm_torch,
    B_glasso = NULL,
    init = init,
    init_scale = 0.02,
    seed = 123,
    d1 = d1, d2 = d2,
    a = a_torch, Z = Z_torch, Theta = Theta, A = A_torch, Kc = Kc_torch, sigmaA = sigmaA,
    rho = best_rho, R = R, Sigma.z = Sigma.z,
    adam_lr = adam_lr,
    adam_steps = adam_steps,
    clip_norm = 1.0,
    lbfgs_lr = lbfgs_lr,
    lbfgs_history_size = 10L,
    lbfgs_max_iter = lbfgs_max_iter,
    lbfgs_tol_abs = 1e-6,
    lbfgs_tol_rel = 1e-4,
    verbose = TRUE
  )
  
  return(list(d_opt = res,
              preProcValues = preProcValues,
              d_covars = d_covars,
              nuis_vars = nuis_vars,
              nuisance = nuisance,
              params = params,
              best_rho = best_rho,
              Vmean = Vmean,
              # all.converge = all.converge,
              train_list = train_list))
}


evalIFTR_L2 <- function(fit, test){
  
  # B <- fit$d_opt$B
  B <- fit$d_opt$best_B
  params <- fit$params
  device <- params$device
  train_list <- fit$train_list
  preProcValues <- fit$preProcValues
  d_covars <- fit$d_covars
  nuisance <- fit$nuisance
  nuis_vars <- fit$nuis_vars
  Kc_nuis <- switch(nuis_vars,
                   "WXZ" = train_list$K_wxz,
                   "WX" = train_list$K_wx,
                   "UX" = train_list$K_ux,
                   "X" = train_list$K_x)
  t_seq <- seq(0,1, length = ncol(train_list$treatment))
  
  # standardize the test data
  test$C_std <- predict(preProcValues, test$C)
  
  # calculate the IFTR in test data
  Theta <- torch_tensor(train_list$Theta, device = device, dtype = torch_float64())
  Z.test <- gen_basisL2(params, data=test$C_std[,column_list[[d_covars]]], type=params$ITR_type, 
                        k.beta=params$k.beta, t_seq = t_seq, train_basis_info = train_list$basis_info)$Z
  d.test <- torch_tensor(Z.test, device = device, dtype = torch_float64())$
    mm(B)$mm(Theta$t())
  KdA <- rbf_kernel_gram_mm(
    d.test,
    torch_tensor(train_list$treatment, device = device,dtype = torch_float64()),
    sigma = as.numeric(params$sigma2)
    )

  # calculate the Kc_testtrain and a.train
  switch(nuisance,
         "KRR" = {
           Kc <- train_list$K_ux
           vars <- c("U", "X1", "X2")
           params$select_lam <- params$krr_lam_orc
         },
         "prox-pmmr" = {
           Kc <- train_list$K_wx
           vars <- c("W", "X1", "X2")
         }
  )
  
  V_est <- function(vars, Kc){
    a.train <- nuis_cf(train_list, params, cf = 1, nuisance = nuisance, Kc=Kc)$a[[1]]
    a.train <- torch_tensor(a.train,device = device, dtype = torch_float64())
    # calculate estimated value
    Kc_testtrain <- G.cont(cont1=test$C_std[,vars],
                           cont2=train_list$C_std[,vars],
                           type=params$ker1,
                           sigma=get_median_s(train_list$C_std[,vars]))
    value.test <- torch_tensor(Kc_testtrain, device = device, dtype = torch_float64())$
      mul(KdA)$mm(a.train)$mean()
    return(value.test)
  }
  value.test <- V_est(vars=vars, Kc=Kc)
  value.emp <- V_est(vars=column_list[[nuis_vars]], Kc=Kc_nuis)
  ITR.mise <- apply(as.array(d.test) - test$f_opt, 1, function(x) trapz(t_seq, x^2))
  vd.test <- mean(1.2*test$C[,'U'] + test$C[,c('X1','X2')] %*% c(0.8, 0.8)  - 10*ITR.mise)
  
  return(list(value.test = as.array(value.test),
              vd.test = vd.test,
              value.emp = as.array(value.emp),
              ITR.mise = as.array(mean(ITR.mise))))
}
