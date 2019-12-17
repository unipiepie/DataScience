install.packages("rpart")
install.packages("rpart.plot")
library(rpart)
library(rpart.plot)
library(xlsx)

#read data files
titanic.train <- read.csv(file = "C:/Users/tatapie/Desktop/Zichao Tian/Titanic train.csv", stringsAsFactors = FALSE, header = TRUE)
titanic.test <- read.csv(file = "C:/Users/tatapie/Desktop/Zichao Tian/Titanic test.csv", stringsAsFactors = FALSE, header = TRUE)

#Processing data
titanicdata <- titanic.train[,c(-1,-4,-9,-11)]
titanictest <- titanic.test[,c(-1,-3,-8,-10)]
titanicdata <- na.omit(titanicdata)
titanictest <- na.omit(titanictest)
head(titanicdata)
head(titanictest)

#Implement decision tree
dtree<-rpart(Survived~., data=titanicdata, method = "class", parms=list(split="information"))

#Draw the decision tree
rpart.plot(dtree,branch=1,type=2, fallen.leaves=T,cex=0.8, sub="Decision tree of Titanic Train")

#predict using Titanictest
pred.tree<-predict(dtree,newdata=titanictest,type="class")

summary(pred.tree)

