
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
  
  # compute the initial beta values
  train_list$b <- calc_initial_b(A=train_list$treatment, Z=train_list$Z, Theta=train_list$Theta,
                                 device = device, ITR_type = params$ITR_type)
  
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
  Theta <- torch_tensor(bases$Theta, device = device)
  valueCV <- matrix(NA, nrow = length(rhos), ncol = K)
  convergeM <- matrix(NA, nrow = length(rhos), ncol = K)
  
  for (i in 1:K){
    cat("\nCV ", i, " start.")
    # use data excluding fold i to fit nuisance functions
    trainidx <- which(cv_idx != i)
    # get train data
    cv_train_i <- list(treatment = train_list$treatment[trainidx,],
                       Z = train_list$Z[trainidx,],
                       K_wx = train_list$K_wx[trainidx,trainidx],
                       K_zx = train_list$K_zx[trainidx,trainidx],
                       KA = train_list$KA[trainidx,trainidx],
                       Y = train_list$Y[trainidx])
    # use train data to calculate nuisance functions
    nuis_list_i <- nuis_cf(cv_train_i, params, cf = cf, nuisance = nuisance, Kc=Kc[trainidx, trainidx])
    rm(cv_train_i)
    
    # obtain the IFTRs for each rho
    for(j in seq_along(rhos)){
      d_opt <- dL2M(b_init = train_list$b,
                    nuis_list = nuis_list_i,
                    bases = bases,
                    params=params, 
                    rho = rhos[j],
                    n_iteration=n_iteration, 
                    lr= lr, 
                    device = device)
      # calculate the value in the test data
      d.test.i <- torch_tensor(nuis_cv$Z.test[[i]],device = device)$mm(d_opt$B)$mm(torch_t(Theta))
      KdA.i <- rbf_kernel_gram_mm(d.test.i, 
                             torch_tensor(nuis_cv$A.train[[i]], device = device), 
                             sigma = params$sigma2)
      valueCV[j,i] <- as.array(torch_tensor(nuis_cv$Kc[[i]], device = device)$mul(KdA.i)$mm(torch_tensor(nuis_cv$a[[i]], device = device))$mean())
      convergeM[j, i] <- d_opt$converge
      rm(d.test.i, KdA.i)
    }
  }
  
  # find the rho that maximize average values
  Vmean <- apply(valueCV, 1, mean)
  max_index <- max(which(Vmean == max(Vmean))) # in case there are multiple max values
  best_rho <- rhos[max_index]
  all.converge <- all(convergeM)
  cat("The best rho is ", best_rho)
  
  # get the final IFTR using the best rho
  d_opt <- dL2M(b_init = train_list$b,
                nuis_list = nuis_cv,
                bases = bases,     
                params = params, 
                rho = best_rho,
                n_iteration=n_iteration, 
                lr= lr, 
                device = device)
  
  return(list(d_opt = d_opt,
              preProcValues = preProcValues,
              d_covars = d_covars,
              nuis_vars = nuis_vars,
              nuisance = nuisance,
              params = params,
              best_rho = best_rho,
              Vmean = Vmean,
              all.converge = all.converge,
              train_list = train_list))
}


evalIFTR_L2 <- function(fit, test){
  
  B <- fit$d_opt$B
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
  Theta <- torch_tensor(train_list$Theta, device = device)
  Z.test <- gen_basisL2(params, data=test$C_std[,column_list[[d_covars]]], type=params$ITR_type, 
                        k.beta=params$k.beta, t_seq = t_seq, train_basis_info = train_list$basis_info)$Z
  d.test <- torch_tensor(Z.test,device = device)$mm(B)$mm(torch_t(Theta))
  KdA <- rbf_kernel_gram_mm(d.test, 
                            torch_tensor(train_list$treatment, device = device), 
                            sigma = params$sigma2)
  
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
    a.train <- torch_tensor(a.train,device = device)
    # calculate estimated value
    Kc_testtrain <- G.cont(cont1=test$C_std[,vars], 
                           cont2=train_list$C_std[,vars], 
                           type=params$ker1, 
                           sigma=get_median_s(train_list$C_std[,vars]))
    value.test <- torch_tensor(Kc_testtrain, device = device)$mul(KdA)$mm(a.train)$mean()
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
