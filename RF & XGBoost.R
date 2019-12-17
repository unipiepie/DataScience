install.packages("randomForest")
install.packages("xgboost")
install.packages("Metrics")
Sys.setenv(JAVA_HOME='C:/Program Files (x86)/Java/jre1.8.0_144')
library(rJava)
library(xlsx)
library(randomForest)
library(xgboost)
library(Metrics)

data <- read.xlsx("E:/study/ÑÐ¶þÉÏ 2018.9-2018.12/7390 Advances Data Sci/homework/hw5/Copy of boston.xls", sheetIndex=1)
data <- data.frame(data)
data
nrow(data)
set.seed(71)
test_sub <- sample(nrow(data), (1/10)*nrow(data))
train_data <- data[-test_sub,]
test_data <- data[test_sub,]
train_data

#Random Forest
data.rf <- randomForest(MV ~ ., data=train_data, importance=TRUE, proximity=TRUE)
print(data.rf)
predict.rf <- predict(data.rf, subset(test_data,select=-c(MV)))
summary(predict.rf)
rmse.rf <- rmse(as.matrix(test_data["MV"]), as.matrix(predict.rf))
rmse.rf
#sqrt(mean((as.matrix(test_data["MV"]) - as.matrix(predict.rf))^2))

#XGBoost
train_matrix <- xgb.DMatrix(data = as.matrix(train_data[!names(train_data) %in% c("MV")]), label = train_data$MV)
data.xgb = xgboost(data=train_matrix, max_depth=3, eta = 0.2, nthread=3, nrounds=40, lambda=0
                     , objective="reg:linear")
predict.xgb <- predict(data.xgb, data.matrix(subset(test_data,select=-c(MV))))
summary(predict.xgb)
rmse.xgb <- rmse(as.matrix(test_data["MV"]), as.matrix(predict.xgb))
rmse.xgb
