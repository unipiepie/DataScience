install.packages("randomForest")
install.packages("xgboost")
install.packages("Metrics")
library(rJava)
library(xlsx)
library(randomForest)
library(xgboost)
library(Metrics)

data <- read.xlsx("C:/Users/tatapie/Desktop/Zichao_Tian/Copy of boston.xls", sheetIndex=1)
head(data)
nrow(data)
set.seed(71)
test_sub <- sample(nrow(data), (1/10)*nrow(data))
test_data <- data[test_sub,]
train_data <- data[-test_sub,]
head(train_data)

#Random Forest
data.rf <- randomForest(MV ~ ., data=train_data, importance=TRUE, proximity=TRUE)
print(data.rf)
prediction.rf <- predict(data.rf, subset(test_data,select=-c(MV)))
summary(prediction.rf)
rmse.rf <- rmse(as.matrix(test_data["MV"]), as.matrix(prediction.rf))
rmse.rf


#XGBoost
train_matrix <- xgb.DMatrix(data = as.matrix(train_data[!names(train_data) %in% c("MV")]), label = train_data$MV)
data.xgb = xgboost(data=train_matrix, max_depth=3, eta = 0.2, nthread=3, nrounds=40, lambda=0
                   , objective="reg:linear")
predict.xgb <- predict(data.xgb, data.matrix(subset(test_data,select=-c(MV))))
summary(predict.xgb)
rmse.xgb <- rmse(as.matrix(test_data["MV"]), as.matrix(predict.xgb))
rmse.xgb
