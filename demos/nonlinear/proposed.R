library(pracma) 
library(splines)
library(MASS)
library(caret)
library(fda)

source("util.R")
source("optim_ms.R")
source("nuisance.R")
source("IFTR_mulstart.R")
source("sim_data.R")

## set the training parameters
seed <- 100
n.train <- 5000
n.test <- 10000
JITTER_W <- 0
JITTER_L <- 0
device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")


## set the parameters in optimization
n_starts <- 3L
adam_steps <- 300L
adam_lr <- 3e-4
init <- "normal"
lbfgs_max_iter <- 200L
lbfgs_lr <- 0.5
calc_b0 <- F

#### set the parameters
# k1 is the number of B-spline basis used for each variables
# k.beta is the number of B-spline basis functions used to expand the coefficient function
# rho is the penalty used for estimating IFTR
# set ITR_type = "nonlinear" to estimate nonlinear IFTR

params <- list(ker1 = "gauss",
               ker2 = "gauss",
               ITR_type = "nonlinear",
               k1 = 4,
               k.beta = 4,
               rhos = c(10^seq(-13, 0, length=14)),   
               nystr = T,
               device = device,
               M = 1000)           

# parameters used to tune the nuisance functions
tune_args <- list(crit = "lmo", 
                  m = 1, 
                  bl_interval = c(1, 16),
                  folds = NULL)

## generate the train data and use it to estimate the IFTR
# N1 scenario: regime_type = "nonlinear", u_in_f = T
# N2 scenario: regime_type = "nonlinear", u_in_f = F

train <- ProxData2(seed=seed, n=n.train, regime_type = "nonlinear", u_in_f = T)
dPROX <- fitIFTRL2(train, nuisance="prox-pmmr", nuis_vars="WX", d_covars="ZX", params, tune_args,
                 K=3, cf=3)

# generate the test data under the same scenario and obtain the estimated value
test <- ProxData2(seed=10000, n=n.test, regime_type = "nonlinear", u_in_f = T)
vprox <- evalIFTR_L2(dPROX, test)

vprox$vd.test

