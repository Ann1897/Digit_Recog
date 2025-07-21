library(R.matlab)
library(here)
library(tidyverse)

mat_data <- readMat(here("mnist_all.mat"))


##-------Creating Train and Test Datasets---------------------------------------

df <- mat_data

for (i in 1:length(df)) {
  # Get the name of the current matrix
  matrix_name <- names(df)[i]
  
  # Get the current matrix
  current_matrix <- df[[i]]
  
  # Create a new column with the matrix name
  new_column <- rep(matrix_name, nrow(current_matrix))
  
  # Bind the new column to the current matrix
  updated_matrix <- cbind(current_matrix, new_column)
  
  # Update the matrix in the list
  df[[i]] <- updated_matrix
}



DF <- df[[1]]

for (i in 2:20) {
  
  DDF <- rbind(DF, df[[i]])
  DF <- DDF
  
}

DF <- as.data.frame(DF)


summary(as.factor(DF$new_column))

DF$Y <- str_sub(DF$new_column, -1)
DF$Set <- str_sub(DF$new_column, end = -2)


summary(as.factor(DF$Y))
summary(as.factor(DF$Set))

DF_1 <- subset(DF, select = -c(new_column))

Train_df <- DF_1[DF_1$Set == 'train', ]
Test_df <- DF_1[DF_1$Set == 'test', ]


Train_df <- subset(Train_df, select = -c(Set))
Test_df <- subset(Test_df, select = -c(Set))

class(Train_df$Y)


summary(Train_df$Y)
summary(Test_df$Y)


#--------Adjusting the Values ----------------

for (i in 1:(ncol(Train_df)-1)) {
  
  Train_df[[i]] <- as.numeric(Train_df[[i]])
  
}



Train_df_X <- Train_df[,1:784]/255
Train_df <- cbind(Train_df_X, Train_df[,785])

colnames(Train_df)[colnames(Train_df) == "Train_df[, 785]"] <- "Number"


Maxi <- matrix(0, 784, 1)

for (i in 1:784) {
  Maxi[i,1] <- max(Train_df[,i])
  
}

summary(Maxi)


##---------Functions-------------------------------------------------------------


softmax <- function(x) {
  S_exp <- rowSums(exp(x))
  S_exp <- replicate(ncol(x), S_exp)
  return((exp(x)/S_exp))
}


Layer_1 <- function(x, W, b) {
  
  # x is the Train dataset with only predictors with dimensions n*784
  # W is the weight matrix of dimension 784 * 10
  # b is the beta0 weights with dimensions 10*1
  
  O_i <- (as.matrix(x) %*% W)
  biases <- replicate(nrow(O_i), b)
  O_i <- O_i + t(biases)
  Y <- softmax(O_i)
  
  return(Y)
}

# cost function
cost <- function(y, y_) {
  #y is the predicted matrix 
  #y_ is the true values which is a one_hot_matrix
  
  NLL <- matrix(0, nrow(y), 1)
  
  for (i in 1:nrow(y)){
    
    NLL[i] <- (-log((y[i,]) %*% y_[i,]))
    
  }
  
  return(NLL)
}


compute_cost_and_gradient <- function(x, y, W, b ) {
  
  # x is the Train dataset with only predictors with dimensions n*784
  # y is the True value (number) in dimension n*1
  # W is the weight matrix of dimension 784 * 10
  # b is the beta0 weights with dimensions 10*1
  
  #one_hot_matrix
  
  Y_hat <- Layer_1(x, W, b)
  
  Y_H <- matrix(0, nrow = nrow(y), ncol = ncol(Y_hat))
  
  for (i in 1:nrow(y)) {
    Y_H[i, as.numeric(y[i]) + 1] <- 1
  }
  
  # Forward pass
  
  #Y_hat is matrix of dimension n*10
  
  # Negative log-likelihood
  
  nll <- cost(Y_hat, Y_H)
  
  # Backward pass for gradients
  d_logits <- (Y_hat - Y_H)
  grad_W <- t(x) %*% d_logits
  grad_b <- colSums(d_logits)
  
  return(list(nll = nll, grad_W = grad_W, grad_b = grad_b))
  
}




##----------------Weight Initialization------------------------------------------


set.seed(2024)

X_ <- Train_df_X[1:50, ] %>% as.matrix()

Y_ <- Train_df[1:50,785] %>% as.matrix()

# Number of input features (neurons)
num_inputs <- ncol(X_)

# Number of output neurons
num_outputs <- 10

# Initialize weights randomly (e.g., uniform distribution)
weights <- matrix(runif(num_inputs * num_outputs), nrow = num_inputs, ncol = num_outputs)

# Initialize biases (zeros)
biases <- rep(0, num_outputs)

##---------Output---------------------------------------------------------------


compute_cost_and_gradient(X_, Y_, weights, biases)

