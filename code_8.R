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


##---Part--7---------------------------------------------------------------------

library(reticulate)
use_python("C:/Users/annsh/AppData/Local/Programs/Python/Python312/python.exe", required = TRUE)

library(tensorflow)
tf <- import("tensorflow")
keras <- tf$keras

library(keras)



network %>% compile(
  optimizer = "rmsprop",             # network will update itself based on the training data & loss
  loss = "categorical_crossentropy", # measure mismatch between y_pred and y, calculated after each minibatch
  metrics = c("accuracy")            # measure of performace - correctly classified images
)



Train_df_y <- to_categorical(Train_df_y) # makes key-value boolean dummy vars out of numerical vectors
test_labels <- to_categorical(test_labels)   # do the same with the test labels


model <-  keras_Sequential(
  layers.Dense(2, activation="relu", name="layer1"),
  layers.Dense(3, activation="relu", name="layer2"),
  layers.Dense(4, name="layer3") 
  )

