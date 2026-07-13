library(torch)
library(mgcv) # function tensor.prod.model.matrix

#++++++++++++++++++++++++++ Nystroem approximation and matrix inverse ++++++++++++++++++++++++++#

nystroem_decomp_torch <- function(K, ind){
  Knm <- K[,ind]
  Kmm <- Knm[ind,]
  Kmm <- 0.5 * (Kmm + torch_t(Kmm))
  eig_decomp <- torch::linalg_eigh(Kmm)
  
  eig_val <- eig_decomp[[1]]
  eig_vec <- eig_decomp[[2]]
  
  eig_val_clamped <- torch_clamp(eig_val, min = 1e-5)
  U <-  sqrt(length(ind)/nrow(K)) * Knm$mm(eig_vec)$mm(torch_diag(1/eig_val_clamped))
  V <- torch_diag(eig_val) * nrow(K)/length(ind)
  
  return(list(U = U, V = V))
}

# stable, high precision and fast speed
fast_pinv <- function(A_torch){

  # SVD decomposition
  svd_result <- linalg_svd(A_torch)
  U <- svd_result[[1]]
  S <- svd_result[[2]]
  Vh <- svd_result[[3]]
  
  # Apply threshold to singular values 
  tol <- 1e-12
  S_inv <- torch_where(S > tol, 1 / S, torch_zeros_like(S))
  
  # Reconstruct pseudo-inverse
  A_pinv_torch <- torch_t(Vh)$mm(torch_diag(S_inv))$mm(torch_t(U))
  A_pinv_torch
}

#+++++++++++++++++++++++++++++++++++++ Gram matrix +++++++++++++++++++++++++++++++++++++#
#+

# function for median heuristic for scalar
get_median_s <- function(X){
  pairwise_dist <- dist(X, diag=F, method= "euclidean")^2
  median <- sqrt(median(pairwise_dist))
  return(median)
}


get_bw_A <- function(X) {
  dist_sq <- torch_sum((X$unsqueeze(2) - X$unsqueeze(1))^2, dim = 3)$multiply(1/dim(X)[2])
  bw_sq <- dist_sq[lower.tri(dist_sq, diag = FALSE)]$median()
  bw <- sqrt(as.array(bw_sq))
  return(bw)
}

G.cont <- function(cont1, cont2=NULL, type="gauss", sigma=1){
  
  if(is.null(dim(cont1))){cont1 <- matrix(cont1, nrow=1)}

  if (type=="gauss"){
    if(is.null(cont2)){
      dist.cont <- as.matrix(dist(cont1, diag=T,upper=T, method= "euclidean")^2)
    }else{
      cont12 <- rbind(cont1, cont2)
      dist.cont.tol <- as.matrix(dist(cont12, diag=T, upper=T, method= "euclidean")^2)
      idx1 <- nrow(cont1)
      idx2 <- idx1 + 1
      dist.cont <- dist.cont.tol[1:idx1,idx2:nrow(cont12)]
    }
    G <- exp(-dist.cont/(2*sigma^2))
  }else if (type=="sob"){
    if (is.null(cont2)){cont2=cont1}
    G <- matrix(NA, nrow=nrow(cont1), ncol=nrow(cont2))
    for (i in 1:nrow(cont1)){
      for (j in 1:nrow(cont2)){
        G[i,j] <- K.sob.prod(cont1[i,],cont2[j,])
      }
    }
  }else{return('Error in the kernel type.')}
  
  return(G)
}

rbf_kernel_gram <- function(X, Y, sigma) {
  dist_sq <- torch_sum((X$unsqueeze(2) - Y$unsqueeze(1))^2, dim = 3)$multiply(1/dim(Y)[2])
  gram_matrix <- torch_exp(-dist_sq / (2 * sigma^2))
  return(gram_matrix)
}

rbf_kernel_gram_mm <- function(X, Y, sigma) {
  d <- X$size(2)
  X2 <- (X$pow(2))$sum(dim = 2, keepdim = TRUE)         # (n×1)
  Y2 <- (Y$pow(2))$sum(dim = 2, keepdim = TRUE)$t()     # (1×m)
  XY <- X$mm(torch_t(Y))                                # (n×m)
  dist_sq <- X2 + Y2 - 2 * XY                           # (n×m)
  dist_sq <- dist_sq$multiply(1/d)
  dist_sq <- torch_clamp(dist_sq, min=0)                # numerical safety
  torch_exp(-dist_sq / (2 * sigma * sigma))
 }
#+++++++++++++++++++++++++++++++++++++ Generate Basis +++++++++++++++++++++++++++++++++++++#


# Step 1: Function to prepare training basis and formula
build_tensor_basis <- function(data, df = 7) {
  data <- data.frame(data)
  basis_vars <- list()
  basis_info <- list()
  
  # Construct bs() for each variable and store attributes
  for (var in names(data)) {
    bs_obj <- bs(data[[var]], df = df, intercept = TRUE)
    basis_vars[[var]] <- bs_obj
    basis_info[[var]] <- attributes(bs_obj)
  }
  mm_train <- tensor.prod.model.matrix(basis_vars)
  
  list(
    model_matrix = mm_train,
    basis_info = basis_info
  )
}

# Step 2: Function to apply basis to test data using training basis_info
# truncated test data so that the values are within boundaries
apply_tensor_basis <- function(test_data, basis_info) {
  test_data <- data.frame(test_data)
  basis_vars_test <- list()
  
  for (var in names(test_data)) {
    attr <- basis_info[[var]]
    # clamp the test values
    test_data[[var]] <- pmin(pmax(test_data[[var]], attr$Boundary.knots[1]), attr$Boundary.knots[2])
    basis_vars_test[[var]] <- bs(test_data[[var]],
                                 knots = attr$knots,
                                 degree = attr$degree,
                                 Boundary.knots = attr$Boundary.knots,
                                 intercept = TRUE)
  }
  # Generate model matrix for test data
  mm_test <- tensor.prod.model.matrix(basis_vars_test)
  return(mm_test)
}

column_list <- list(
  "X" = c("X1", "X2"),
  "ZX" = c("Z", "X1", "X2"),
  "UX" = c("U", "X1", "X2"),
  "WXZ" = c("W", "X1","X2","Z"),
  "WX" = c("W", "X1","X2")
)

gen_basisL2 <- function(params, data, type="nonlinear",k.beta = 7, t_seq,
                      train_basis_info = NULL){
  # k1 is the number of B-spline used each covariate: x1, x2, z
  # subset the dataset that only contains d_covars 
  df <- params$k1
  data <- as.data.frame(data)
  
  if (type =="linear"){
    Z <- model.matrix(~ ., data = data)
    basis_info <- NULL
  }else if (type =="nonlinear"){
    if(is.null(train_basis_info)){
      nl_basis <- build_tensor_basis(data, df = df)
      Z <- nl_basis$model_matrix
      basis_info <- nl_basis$basis_info
    }else{
      Z <- apply_tensor_basis(data, train_basis_info)
      basis_info <- train_basis_info
    }
  }
  
  basisobj <- create.bspline.basis(c(0,1), nbasis = k.beta)
  Theta <- eval.basis(t_seq, basisobj)
  R <- getbasispenalty(basisobj)
  Sigma.z <- t(Z) %*% Z/nrow(Z)
  
  return(list(Theta = Theta, 
              Z = Z, 
              R = R, 
              q = ncol(Z), 
              Sigma.z = Sigma.z, 
              basis_info = basis_info))
}

