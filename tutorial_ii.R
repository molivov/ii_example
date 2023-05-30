#! tutorial ii

# load required packages
library(methods)
library(utils)
library(stats)
library(optimx)

set.seed(238476)

# load the data
data <- read.csv("data_ii.csv")

X <- as.matrix(data[, c("x1", "x2")])
y <- as.matrix(data["y"])

# draws of normal random numbers
M <- 10
u <- matrix(rnorm(length(y)*M, mean = 0, sd = sqrt(1.5)), ncol=10)

# define auxiliary model estimator
ols <- function(X, y, add.constant=TRUE){
  if (add.constant==TRUE){
    n <- length(y)
    X <- cbind(rep(1, n), X)
  }
  solve(t(X) %*% X) %*% t(X) %*% y
}

# estimate aux model with observed data
beta.hat <- ols(X, y)

# define structural model
G <- function(X, u, theta){
  n <- dim(X)[2]
  exp(cbind(rep(1, n), X) %*% theta) + u
}

# define criterion for ii
wald <- function(beta.tilde, beta.hat){
  sum((beta.tilde - beta.hat)^2)
}

# define objective function
obj <- function(theta, X, u, beta.hat){
  y.tilde <- apply(u, 2, function(u) G(X, u, theta)) # this simulates the data
  beta.tilde <- apply(y.tilde, 2, function(y.tilde) ols(X, y.tilde)) # this estimates aux model on simulated data
  beta.tilde <- apply(beta.tilde, 1, mean) # this averages aux model estimates
  wald(beta.tilde, beta.hat)
}

# optimization of criterion
theta.0 <- as.vector(ols(X, y))
opt <- optimx(theta.0, function(theta) obj(theta, X, u, beta.hat), method="Nelder-Mead")
opt

# the true values are
# 0.9279139553540544
# 0.2408481106524355
# 0.4354059904502885
