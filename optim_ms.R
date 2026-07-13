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

# ------------------------------------------------------------
# Full-batch Adam  ->  Full-batch L-BFGS  (with MULTI-START)
# Loss: valuecfM(B, a, Z, Theta, A, Kc, sigmaA)
# Penalty: rho * tr( B R B^T Sigma.z )
#
# Requirements:
# - torch (R)
# - your function: rbf_kernel_gram_mm(d.test, A[[i]], sigma = sigmaA)
# - Data objects: a, Z, Theta, A, Kc, sigmaA
# ------------------------------------------------------------

library(torch)

device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")

# -----------------------------
# 0) P(B) = tr( B R B^T Sigma )
# -----------------------------
penalty_trace <- function(B, R, Sigma.z) {
  # scalar torch tensor
  torch_trace(B$mm(R)$mm(torch_t(B))$mm(Sigma.z))
}

# -----------------------------
# 1) loss
# -----------------------------
valuecfM <- function(B, a, Z, Theta, A, Kc, sigmaA) {
  cf <- length(a)
  acc <- torch_zeros(1, device = B$device, dtype = B$dtype)
  
  for (i in seq_len(cf)) {
    d.test <- Z[[i]]$mm(B)$mm(Theta$t())
    KdA <- rbf_kernel_gram_mm(d.test, A[[i]], sigma = sigmaA)
    acc <- acc + (Kc[[i]]$mul(KdA)$mm(a[[i]])$mean())
  }
  
  acc / cf
}

# -----------------------------
# 2) Full objective: loss + rho * penalty
# -----------------------------
objective_full <- function(B, a, Z, Theta, A, Kc, sigmaA,
                           rho, R, Sigma.z) {
  value <- valuecfM(B, a, Z, Theta, A, Kc, sigmaA)
  pen  <- penalty_trace(B, R, Sigma.z)
  - value + rho * pen
}

# -----------------------------
# 3) One run: Full-batch Adam -> L-BFGS
# -----------------------------
run_one_start <- function(
    B0,
    a, Z, Theta, A, Kc, sigmaA,
    rho, R, Sigma.z,
    adam_lr = 3e-4,
    adam_steps = 200L,
    clip_norm = 1.0,
    lbfgs_lr = 0.5,
    lbfgs_history_size = 10L,
    lbfgs_max_iter = 200L,
    lbfgs_tol_abs = 1e-6,
    lbfgs_tol_rel = 1e-4,
    verbose = TRUE
) {
  # Make trainable copy
  B <- B0$clone()$detach()$requires_grad_(TRUE)
  
  # ---- Full-batch Adam phase ----
  adam <- optim_adam(list(B), lr = adam_lr)
  
  for (t in seq_len(adam_steps)) {
    adam$zero_grad()
    
    obj <- objective_full(
      B, a, Z, Theta, A, Kc, sigmaA,
      rho, R, Sigma.z
    )
    
    obj$backward()
    
    # Gradient clipping
    if (!is.null(clip_norm) && is.finite(clip_norm) && clip_norm > 0) {
      nn_utils_clip_grad_norm_(list(B), max_norm = clip_norm)
    }
    
    adam$step()
    
    if (verbose && (t %% 20L == 0L)) {
      with_no_grad({
        val <- valuecfM(B, a, Z, Theta, A, Kc, sigmaA)$item()
        pen <- penalty_trace(B, R, Sigma.z)$item()
      })
      cat(sprintf("  Adam %d/%d | obj=%.6f | value=%.6f | pen=%.6f\n",
                  t, adam_steps, as.numeric(obj$item()), as.numeric(val), as.numeric(pen)))
    }
  }
  
  # ---- L-BFGS phase ----
  lbfgs <- optim_lbfgs(
    list(B),
    lr = lbfgs_lr,
    max_iter = lbfgs_max_iter,
    history_size = lbfgs_history_size,
    line_search_fn = "strong_wolfe",
    tolerance_grad = 1e-7,
    tolerance_change = 1e-9
  )
  
  # Track L-BFGS progress
  lbfgs_iter <- 0
  lbfgs_obj_history <- numeric(0)
  
  # Closure returns the loss tensor for L-BFGS line search
  closure <- function() {
    lbfgs$zero_grad()
    obj <- objective_full(B, a, Z, Theta, A, Kc, sigmaA, rho, R, Sigma.z)
    obj$backward()
    
    # Track objective values
    lbfgs_iter <<- lbfgs_iter + 1
    lbfgs_obj_history <<- c(lbfgs_obj_history, as.numeric(obj$item()))
    
    return(obj)
  }
  
  # Run L-BFGS
  lbfgs$step(closure)
  
  # Final evaluation (no grad) - this is the TRUE final objective
  with_no_grad({
    final_obj <- objective_full(B, a, Z, Theta, A, Kc, sigmaA, rho, R, Sigma.z)$item()
    final_val <- valuecfM(B, a, Z, Theta, A, Kc, sigmaA)$item()
    final_pen <- penalty_trace(B, R, Sigma.z)$item()
  })
  
  # Assess convergence using absolute and relative change
  converged <- FALSE
  convergence_reason <- "max_iter_reached"
  
  # Use last few iterations to check stability
  recent_objs <- tail(lbfgs_obj_history, min(10, length(lbfgs_obj_history)))
  abs_change <- max(abs(diff(recent_objs)))
  rel_change <- abs_change / (abs(mean(recent_objs)) + 1e-10)
  
  # Check convergence
  if (abs_change < lbfgs_tol_abs) {
    converged <- TRUE
    convergence_reason <- "absolute_change_small"
  } else if (rel_change < lbfgs_tol_rel) {
    converged <- TRUE
    convergence_reason <- "relative_change_small"
  }
  
  if (verbose) {
    cat(sprintf("  LBFGS complete | obj=%.8f | value=%.8f | pen=%.8f\n",
                as.numeric(final_obj), as.numeric(final_val), as.numeric(final_pen)))
    cat(sprintf("  Converged: %s (%s) | %d calls | abs_chg=%.2e | rel_chg=%.2e\n",
                converged, convergence_reason, lbfgs_iter, abs_change, rel_change))
  }
  
  list(
    B = B$detach(),
    obj = as.numeric(final_obj),
    value = as.numeric(final_val),
    pen = as.numeric(final_pen),
    converged = converged,
    convergence_reason = convergence_reason,
    lbfgs_iters = lbfgs_iter
  )
}

# -----------------------------
# 4) MULTI-START 
# -----------------------------
multi_start_opt <- function(
    n_starts = 10L,
    B_warm = NULL,
    B_glasso = NULL,
    init = c("normal", "zeros"),
    init_scale = 0.05,
    seed = 123,
    d1, d2,
    # data/loss stuff
    a, Z, Theta, A, Kc, sigmaA,
    # penalty stuff
    rho, R, Sigma.z,
    # optimizer settings
    adam_lr = 3e-4,
    adam_steps = 200L,
    clip_norm = 1.0,
    lbfgs_lr = 0.5,
    lbfgs_history_size = 10L,
    lbfgs_max_iter = 200L,
    lbfgs_tol_abs = 1e-7,
    lbfgs_tol_rel = 1e-7,
    verbose = TRUE
) {
  init <- match.arg(init)
  torch_manual_seed(seed)
  
  best_obj <- Inf
  best_B <- NULL
  all_obj <- numeric(n_starts)
  all_val <- numeric(n_starts)
  all_converged <- logical(n_starts)
  all_convergence_reason <- character(n_starts)
  
  for (s in seq_len(n_starts)) {
    
    if (!is.null(B_warm) && s == 1L) {
      # ---- Warm start ----
      B0 <- B_warm$clone()$detach()
      if (verbose) cat("\nStart 1 (warm start)\n")
    } else if (!is.null(B_glasso) && s == 2L) {
      # ---- start with group lasso coefficients ----
      B0 <- init_scale * B_glasso$clone()$detach()
      if (verbose) cat("\nStart 2 (glasso start)\n")
    } else {
      # ---- Random start ----
      if (init == "zeros") {
        B0 <- torch_zeros(d1, d2, device=device, dtype=torch_float64())
      } else {
        B0 <- init_scale * torch_randn(d1, d2, device=device, dtype=torch_float64())
      }
      if (verbose) cat(sprintf("\nStart %d/%d (random)\n", s, n_starts))
    }
    
    res <- run_one_start(
      B0,
      a, Z, Theta, A, Kc, sigmaA,
      rho, R, Sigma.z,
      adam_lr = adam_lr,
      adam_steps = adam_steps,
      clip_norm = clip_norm,
      lbfgs_lr = lbfgs_lr,
      lbfgs_history_size = lbfgs_history_size,
      lbfgs_max_iter = lbfgs_max_iter,
      lbfgs_tol_abs = lbfgs_tol_abs,
      lbfgs_tol_rel = lbfgs_tol_rel,
      verbose = verbose
    )
    
    all_obj[s] <- res$obj
    all_val[s] <- res$value
    all_converged[s] <- res$converged
    all_convergence_reason[s] <- res$convergence_reason
    
    if (verbose) {
      cat(sprintf("  -> Start %d final obj = %.8f | final value= %.8f | converged=%s (%s)\n", 
                  s, res$obj, res$value, res$converged, res$convergence_reason))
    }
    
    if (res$obj < best_obj) {
      best_obj <- res$obj
      best_B <- res$B
    }
  }
  
  list(
    best_B = best_B, 
    best_obj = best_obj, 
    objs = all_obj, 
    vals = all_val,
    converged = all_converged,
    convergence_reasons = all_convergence_reason
  )
}
# ============================================================

calc_initial_b <- function(A, Z, Theta, device, ITR_type){
  if (ncol(Z) > 200){
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
    
  }else{
    y <- as.vector(t(A))
    ZkronT <- kronecker(Z, Theta)
    if(is_singular(t(ZkronT) %*% ZkronT)){
      y <- torch_tensor(y, device = device)
      ZkronT <- torch_tensor(kronecker(Z, Theta), device = device)
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

