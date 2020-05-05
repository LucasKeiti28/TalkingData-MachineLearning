getwd()
dir()

# Business's problem: Predict whether a user will download an app after clicking a mobile app advertisement.
# Machine Learning - supervisioned analisys - classification

# Loading data
require(readr)
require(dplyr)
?read_csv2
?read.csv
?dplyr
df_train <- read.csv("train_sample.csv", sep=",")

# View data
View(df_train)
str(df_train)
summary(df_train)
dim(df_train)

df_train_downloaded <- df_train %>%
  filter(is_attributed == 1)

downloaded <- nrow(df_train_downloaded)
downloaded_rate <- downloaded/nrow(df_train) * 100

# Transforming numerical variables in factors (categorical variables)
df_train$ip_df <- as.factor(df_train$ip)
df_train$app_df <- as.factor(df_train$app)
df_train$device_df <- as.factor(df_train$device)
df_train$os_df <- as.factor(df_train$os)
df_train$channel_df <- as.factor(df_train$channel)
df_train$is_attributed_df <- as.factor(df_train$is_attributed)

# Creating new dataset to balanced target variable (is_attributed_df).
df_train_balanced <- df_train_downloaded
# Selecting randomly 227 observations which is_attributted_df == 0.
df_train_not_downloaded <- df_train %>%
  filter(is_attributed == 0)
df_train_not_downloaded <- df_train_not_downloaded[sample(nrow(df_train_not_downloaded), 227), ]

head(df_train_downloaded)
head(df_train_not_downloaded)

# Combine downloaded and not_downloaded observation in one dataset.
df_train_balanced <- bind_rows(df_train_downloaded, df_train_not_downloaded)
View(df_train_balanced)

# Exploratory Analysis
str(df_train_balanced)
summary(df_train_balanced)

require(ggplot2)
ggplot(df_train_balanced, aes(device_df)) + geom_bar() + ggtitle("Devices vs. Quantity")

# Using random forest to make features selection
require(randomForest)
model <- randomForest( is_attributed_df ~ ip + app +device + os + channel, data = df_train_balanced,
                       ntree = 100, nodesize = 10, importance = T)
varImpPlot(model)

# the most significative variables are app, channel and ip.

# Creating ML Model, using randomforest
nrow(df_train)
nrow(df_train_balanced)

model <- randomForest( is_attributed_df ~ ip + app + channel, data = df_train_balanced,
                       ntree = 100, nodesize = 10)
print(model)

# Model's score
predictions <- predict(model, df_train)
predictions <- as.data.frame(predictions)

df_train <- bind_cols(df_train, predictions)
correct_predictions <- nrow(df_train %>%
                              filter(is_attributed_df == predictions))
correct_rate <- correct_predictions/nrow(df_train) * 100
print(correct_rate)

# Evaluate model
require(caret)
df_matrix <- data.frame(df_train$is_attributed_df, df_train$predictions)
str(df_matrix)
confusionMatrix(df_train$predictions, reference = df_train$is_attributed_df)

#Accuracy : 0.9497  

require(ROCR)
class1 <- predict(model, newdata = df_train, type="prob")
class2 <- df_train$is_attributed_df

pred <- prediction(class1[,2], class2)
perf <- performance(pred, "tpr", "fpr")
plot(perf, col = rainbow(10))

# Applying the model created to testing data
df_test <- read.csv("test.csv", sep=",")
head(df_test)
dim(df_test)
predictions_test <- predict(model, df_test)
predictions_test <- as.data.frame(predictions_test)
summary(predictions_test)

# Generate dataset with results predicted
?bind_cols
click_id <- as.data.frame(df_test$click_id)
df_result <- bind_cols(click_id, predictions_test)
summary(df_result)
write_csv(df_result, "talkingdata-result.csv")

