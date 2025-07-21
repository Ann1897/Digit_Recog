library(R.matlab)
library(here)
library(tidyverse)

mat_data <- readMat(here("mnist_all.mat"))


df <- mat_data

mat_data$train6[15,]
mat_data$train7[20,]
mat_data$train8[40,]
mat_data$train9[11,]


show_digit <- function(arr784, col=gray(12:1/12), ...) {
  image(matrix(arr784, nrow=28)[,28:1], col=col, ...)
}

show_digit(mat_data$train0[9, ], asp = 1)





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


##----------------Weight Initialization------------------------------------------

# Number of input features (neurons)
num_inputs <- 784

# Number of output neurons
num_outputs <- 10

# Initialize weights randomly (e.g., uniform distribution)
weights <- matrix(runif(num_inputs * num_outputs), nrow = num_inputs, ncol = num_outputs)


# Initialize biases (zeros)
biases <- rep(0, num_outputs)

##----------Initial_Algorithnm--------------------------------------------------

#Subset

M_0 <- subset(Train_df, Train_df$Number == 0 )

M_0_1 <- M_0[,-785]

O_i <- as.matrix(M_0_1) %*% as.matrix(weights[,1])

O_i <- (O_i + biases[1])

softmax <- function(x) {
  exp_x <- exp(x)
  softmax_values <- exp_x / sum(exp_x)
  return(softmax_values)
}

M_0_1 <- cbind(M_0, O_i)

M_0_1$'Y' <- softmax(M_0_1$O_i) 


F_Train <- M_0_1


for (i in 1:9) {
  
  M_i <- subset(Train_df, Train_df$Number == i )
  
  M_i_1 <- M_i[,-785]
  
  O_i <- as.matrix(M_i_1) %*% as.matrix(weights[,(i+1)])
  
  O_i <- (O_i + biases[(i+1)])
  
  M_i_1 <- cbind(M_i, O_i)
  
  M_i_1$'Y' <- softmax(M_i_1$O_i) 
  
  F_Train <- rbind(F_Train, M_i_1)
  
}





F_Train %>%
  group_by(Number) %>%
  summarise(Sum = sum(Y), n = n(), sd = sd(Y), Maximum = max(Y), Minimum = min(Y))


