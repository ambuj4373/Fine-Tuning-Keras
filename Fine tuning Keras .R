# Building a Diabetes Classification Model with Keras and R
# --------------------------------------------------------

# Step 1: Load the required libraries
library(keras)
library(caret)

# Step 2: Read the dataset
df <- read.csv("~/Desktop/diabetes 2.csv")

# Step 3: Split the dataset into features and target variable
features <- df[, 1:8] # Extract the features from the dataset
target <- df[, 9] # Extract the target variable from the dataset

# Step 4: Scale the features
scaled_features <- scale(features) # Standardize the features by scaling them

# Step 5: Split the data into train and test sets
set.seed(123) # Set a seed for reproducibility
train_indices <- createDataPartition(target, p = 0.7, list = FALSE) # Split the indices for training set
train_features <- scaled_features[train_indices, ] # Extract the scaled features for training set
train_target <- target[train_indices] # Extract the target variable for training set
test_features <- scaled_features[-train_indices, ] # Extract the scaled features for test set
test_target <- target[-train_indices] # Extract the target variable for test set

# Step 6: Define the model architecture
model <- keras_model_sequential() # Create a sequential model
model %>%
  # Add a dense layer with 32 units and ReLU activation function
  layer_dense(units = 32, activation = "relu", input_shape = c(8)) %>%
  # Add a dense layer with 1 unit and sigmoid activation function
  layer_dense(units = 1, activation = "sigmoid")

# Step 7: Compile the model
model %>%
  compile(
    loss = "binary_crossentropy", # Use binary cross-entropy as the loss function
    optimizer = "adam", # Use Adam optimizer for model optimization
    metrics = c("accuracy") # Track accuracy as the evaluation metric
  )

# Step 8: Fit the model on the training data
history <- model %>% fit(
  x = train_features, # Training features
  y = train_target, # Training target variable
  epochs = 50, # Number of epochs for training
  batch_size = 32 # Batch size for training
)


# Hyperparameter Tuning using Keras in R
# --------------------------------------------------------

# Load the required libraries
library(keras)
library(kerastuneR)
library(caret)

# Step 1: Read the dataset
df <- read.csv("~/Desktop/diabetes 2.csv")

# Step 2: Split the dataset into features and target variable
features <- df[, 1:8]
target <- df[, 9]

# Step 3: Scale the features
scaled_features <- scale(features)

# Step 4: Split the data into train and test sets
set.seed(123)
train_indices <- createDataPartition(target, p = 0.7, list = FALSE)
train_features <- scaled_features[train_indices, ]
train_target <- target[train_indices]
test_features <- scaled_features[-train_indices, ]
test_target <- target[-train_indices]

# Step 5: Define the model architecture
model <- keras_model_sequential()
model %>%
  layer_dense(units = 32, activation = "relu", input_shape = c(8)) %>%
  layer_dense(units = 1, activation = "sigmoid")

# Step 6: Compile the model
model %>%
  compile(
    loss = "binary_crossentropy",
    optimizer = "adam",
    metrics = c("accuracy")
  )

# Step 7: Fit the model on the training data
history <- model %>% fit(
  x = train_features,
  y = train_target,
  epochs = 50,
  batch_size = 32
)

# Hyperparameter Tuning using Keras Tuner

# Step 1: Define the hyperparameters to tune
hyperparameters <- list(
  units = c(16, 32, 64),                           # Number of units in the dense layer
  activation = c("relu", "sigmoid"),               # Activation function for the dense layer
  learning_rate = c(0.001, 0.01, 0.1),             # Learning rate for the optimizer
  batch_size = c(16, 32, 64)                       # Batch size for training
)

evaluation_metric <- "accuracy"                    # Evaluation metric to track during training

# Step 2: Create the search space by combining hyperparameters
search_space <- expand.grid(
  units = hyperparameters$units,
  activation = hyperparameters$activation,
  learning_rate = hyperparameters$learning_rate,
  batch_size = hyperparameters$batch_size
)

best_accuracy <- 0                                   # Variable to store the best accuracy
best_model <- NULL                                   # Variable to store the best model

# Step 3: Iterate over the search space and train models with different hyperparameters
for (i in 1:nrow(search_space)) {
  
  # Step 4: Create the model with the current hyperparameters
  current_model <- keras_model_sequential()
  current_model %>%
    layer_dense(units = search_space$units[i], activation = search_space$activation[i], input_shape = c(8)) %>%
    layer_dense(units = 1, activation = "sigmoid")
  
  # Step 5: Compile the model
  current_model %>%
    compile(
      loss = "binary_crossentropy",
      optimizer = optimizer_adam(learning_rate = search_space$learning_rate[i]),
      metrics = c(evaluation_metric)
    )
  
  # Step 6: Fit the model on the training data
  current_model %>% fit(
    x = train_features,
    y = train_target,
    epochs = 50,
    batch_size = search_space$batch_size[i],
    verbose = 0
  )
  
  # Evaluate the model on the validation data
  evaluation_result <- current_model %>% evaluate(
    x = test_features,
    y = test_target,
    verbose = 0
  )
  current_accuracy <- evaluation_result[[1]]
  
  # Check if the current model performs better than the previous best model
  if (current_accuracy > best_accuracy) {
    best_accuracy <- current_accuracy
    best_model <- current_model
  }
}

# Evaluate the best model
evaluation_result <- best_model %>% evaluate(
  x = test_features,
  y = test_target,
  verbose = 0
)
accuracy <- evaluation_result[[1]]

# Print the accuracy
cat("Accuracy of the best model:", accuracy, "\n")

# Provide feedback based on the model performance
if (accuracy > 0.8) {
  print("The model performs well.")
} else {
  print("The model needs further improvement.")
}

# Store the results for future reference
results <- list(
  accuracy = accuracy,
  hyperparameters = search_space
)

