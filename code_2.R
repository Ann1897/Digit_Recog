library(R.matlab)
library(here)
library(tidyverse)

mat_data <- readMat(here("mnist_all.mat"))

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

#Train_df$Y <- as.factor(Train_df$Y)
#Test_df$Y <- as.factor(Test_df$Y)

#class(Train_df$Y)

summary(Train_df$Y)
summary(Test_df$Y)


##-----Linear Model-------------------------------------------------------------



for (i in 1:(ncol(Train_df)-1)) {
  
  Train_df[[i]] <- as.numeric(Train_df[[i]])
  
}



Train_df_X <- Train_df[,1:784]/255
Train_df <- cbind(Train_df_X, Train_df[,785])

colnames(Train_df)[colnames(Train_df) == "Train_df[, 785]"] <- "Y"


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

Y_Train <- matrix(0, 60000, 10)

M_i <- subset(Train_df, Train_df$Y == 1 )

M_i <- M_i[,-785]



for (j in 1:10) {
  
  M_i <- subset(Train_df, Train_df$Y == (j-1) )
  M_i <- M_i[,-785]
  
  for (i in 1:nrow(M_i)) {
    Y_Train[i,j] <- (as.matrix(M_i[i,]) %*% as.matrix(weights[,j]))
  }
}


O_i <- gather(as.data.frame(Y_Train), "O_i", "Value", 1:10)

O_i <- O_i[!O_i$Value == 0,]

min(O_i$Value)

summary(as.factor(O_i$O_i))
