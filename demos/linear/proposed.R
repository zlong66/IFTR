library(pracma) 
library(splines)
library(MASS)
library(caret)
library(fda)

source("util.R")
source("optim.R")
source("nuisance.R")
source("IFTR.R")
source("sim_data.R")

## set the training parameters
# k.beta is the number of B-spline basis functions used to expand the coefficient function
seed <- 100
n.train <- 500
n.test <- 10000
k.beta <- 7

device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")

#### set the parameters
# set ITR_type = "linear" to estimate linear IFTR
# rho is the penalty used for estimating IFTR
params <- list(ker1 = "gauss",
               ker2 = "gauss",
               ITR_type = "linear",       
               k.beta = k.beta, 
               rhos = 10^seq(-23, 4, length= 28),    
               nystr = ifelse(n.train > 1500, T, F),
               device = device,
               M = 800)           

if(n.train == 1000){JITTER_LWL <- 2e-7}

## generate the train and test data sets
# L1 scenario: regime_type = "linear", u_in_f = T
# L2 scenario: regime_type = "linear", u_in_f = F

train <- ProxData2(seed=seed, n=n.train, regime_type = "linear", u_in_f = T)
test <- ProxData2(seed=10000, n=n.test, regime_type = "linear", u_in_f = T)

## set parameters range used to tune the parameters in nuisance functions

tune_args <- list(crit = "lmo", 
                  m = 1, 
                  bl_interval = if (n.train < 1500) c(1, 22) else c(0.1, 7),
                  folds = NULL)

# estimate the IFTR using the train data set
dPROX <- fitIFTRL2(train, nuisance="prox-pmmr", nuis_vars="WX", d_covars="ZX", params, tune_args,
                 K=3, cf=3)
# obtain the estimated value using test data set
vprox <- evalIFTR_L2(dPROX, test)
vprox$vd.test

