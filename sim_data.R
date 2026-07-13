

########### prepare the dataset ###########
mu <- c(1, 1)
Sigma <- matrix(c(1, 0.6, 0.6, 1), 2, 2)
t <- seq(0,1, len=100)
kappa <- sin(2*pi*t)
Phi <- cbind(sin(4*pi*t), sin(2*pi*t), cos(2*pi*t), cos(6*pi*t))

ProxData2 <- function(seed, n=1000,
                      regime_type = "nonlinear",
                      u_in_f = T){
  set.seed(seed)
  U <- runif(n, min = 0, max = 4)
  X <- mvrnorm(n=n, mu, Sigma)
  W <- cbind(U,X) %*% c(1, 2, 3) + rnorm(n, sd = 0.5)
  A <- matrix(0, nrow = n, ncol = length(t))
  Ac <- cbind(U, X, rnorm(n, sd=0.1*4))
  c <- ifelse(u_in_f, 1, 0)
  for (k in 1:ncol(Phi)){
    A <- A + Ac[,k] %o% Phi[,k]/4
  }
  basis <- bs(t, df = 7, intercept = T)
  beta1 <- basis %*% c(1,3,-3,3,3,-5,2)/10
  beta2 <- basis %*% c(1,2, 1,0,0,0,0)/10
  beta3 <- basis %*% c(0,0, 1,2,2,0,0)/20
  Phi_f <- cbind(beta1, beta2, beta3)

  if (regime_type == "nonlinear"){
    beta1 <- basis %*% c(1,2,0,0,0,0,0)/5
    beta2 <- basis %*% c(0,0, 1,2,0,0,0)/5
    beta3 <- basis %*% c(0,0, 0,0,-2,0,2)/5
    Phi_f <- cbind(beta1, beta2, beta3)
    
    if (u_in_f) {
      Fc <- cbind(X[,1]*X[,2], (0.6*c)*U*X[,1], (1*c)*U*X[,2])
    } else{
      Fc <- cbind(X[,1]*X[,2], (0.6)*X[,1], X[,2]) 
    }
    
  }else if(regime_type == "linear"){
    Fc <- cbind(X[,1]*1.2, 1.6*X[,2], (1*c)*U)
  }
  f_opt <- matrix(0, nrow = n, ncol = length(t))
  for (k in 1:ncol(Phi_f)){
    f_opt <- f_opt + Fc[,k] %o% Phi_f[,k]
  }
  
  Z <- cbind(U, X) %*% c(1.5, 2, 1.5) + 6*apply(A, 1, function(x) trapz(t, x*kappa)) + 
    rnorm(n, sd = 0.5)
  Y <- 1.2*U + X %*% c(0.8, 0.8)  - 10*
    apply(A-f_opt, 1, function(x) trapz(t, x^2)) +
    rnorm(n, sd = 0.5)
  C <- cbind(Z,X,W,U)
  colnames(C) <- c("Z","X1","X2","W","U")
  
  v.true <- mean(1.2*U + X %*% c(0.8, 0.8))
  
  data <- list(Y=Y,
               treatment=A, 
               C=C,
               f_opt=f_opt, 
               v.true = v.true)
  return(data = data)
}



prepare_data <- function(data, nuisance, params){
  ker1 <- params$ker1; ker2 <- params$ker2
  C_std <- data$C_std
  #### specify the parameters
  KA <- rbf_kernel_gram(X=torch_tensor(data$treatment), 
                        Y=torch_tensor(data$treatment), sigma = params$sigma2)
  KA <- as.array(KA)
  #### calculation of Gram metrices and make data_list
  data_list <- list()
  if(nuisance == "prox-pmmr"|nuisance == "prox-pagmm"){
    data_list$K_wx <- G.cont(C_std[,c("W","X1","X2")], type=ker1, sigma=get_median_s(C_std[,c("W","X1","X2")]))
    data_list$K_zx <- G.cont(C_std[,c("Z","X1","X2")], type=ker1, sigma=get_median_s(C_std[,c("Z","X1","X2")]))
  }
  if(nuisance == "KRR"){
    data_list$K_x <- G.cont(C_std[,c("X1","X2")], type=ker1, sigma = get_median_s(C_std[,c("X1","X2")]))
    data_list$K_ux <- G.cont(C_std[,c("U","X1","X2")], type=ker1, sigma = get_median_s(C_std[,c("U","X1","X2")]))
    data_list$K_wxz <- G.cont(C_std[,c("W","X1","X2","Z")], type=ker1, sigma = get_median_s(C_std[,c("W","X1","X2","Z")]))
  }
  data_list <- c(data_list, 
                 list(Y=data$Y, 
                      treatment=data$treatment, 
                      C_std=C_std, 
                      KA = KA, 
                      f_opt = data$f_opt))
  return(data_list)
}


