install.packages("xgboost")
install.packages("Metrics")

library(rJava)
library(xlsx)
library(xgboost)
library(Metrics)

data <- read.xlsx("C:/Users/tatapie/Desktop/DataScience/HW5/age.xlsx",sheetIndex = 1)
data <- data.frame(data)

nrow(data)
set.seed(71)
test_sub <- sample(nrow(data), (1/9)*nrow(data))
train_data <- data[-test_sub,]
test_data <- data[test_sub,]
train_data



#XGBoost
train_matrix <- xgb.DMatrix(data = as.matrix(train_data[!names(train_data) %in% c("LikesHats")]), label = train_data$LikesHats)
data.xgb = xgboost(data=train_matrix, max_depth=3, eta = 0.2, nthread=3, nrounds=40, lambda=0
                   , objective="reg:linear")
predict.xgb <- predict(data.xgb, data.matrix(subset(test_data,select=-c(LikesHats))))
summary(predict.xgb)
rmse.xgb <- rmse(as.matrix(test_data["LikesHats"]), as.matrix(predict.xgb))
rmse.xgb
