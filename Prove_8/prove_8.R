library(e1071)
library(readr)
library(downloader)
library(rsample)

vdat <- read_csv('vowel.csv')
# dataset 1
# test best gamma and cost
tuned <- tune.svm(as.factor(Class)~., data = train, gamma = 10^(-6:-1), cost = 10^(1:2))
summary(tuned)

## linear, polynomial, radial basis and sigmoid
# build function
vdatp <- function(prop, g, c, ker) 
  {
  data_split <- initial_split(vdat, prop = prop)
  
  train <- training(data_split)
  test  <- testing(data_split)
  test <- data.frame(test)
  
  x <- svm(as.factor(Class)~., data = train, gamma = g, cost = c, kernel = ker)
  prediction <- predict(x, test[,-13])
  
  tab <- table(pred = prediction, true = test[,13])
  tab  
  
  y <- table(prediction == test$Class)
  return(y[2]/(y[1] + y[2]))
}

set.seed(225)
vdatp(.75, 0.1, 10, 'linear')

vdatp(.75, 0.1, 10, 'polynomial')

vdatp(.75, 0.1, 10, 'radial')

vdatp(.75, 0.1, 10, 'sigmoid')

vdatp(.75, 0.01, 10, 'linear')

vdatp(.75, 0.01, 10, 'polynomial')

vdatp(.75, 0.01, 10, 'radial')

vdatp(.75, 0.01, 10, 'sigmoid')

vdatp(.75, 0.1, 5, 'linear')

vdatp(.75, 0.1, 5, 'polynomial')

vdatp(.75, 0.1, 5, 'radial')

vdatp(.75, 0.1, 5, 'sigmoid')

vdatp(.75, 0.01, 5, 'linear')

vdatp(.75, 0.01, 5, 'polynomial')

vdatp(.75, 0.01, 5, 'radial')

vdatp(.75, 0.01, 20, 'sigmoid')





ldat <- read_csv('letters.csv')
data_split <- initial_split(ldat, prop = .7)

train <- training(data_split)
test  <- testing(data_split)
test <- data.frame(test)
tuned <- tune.svm(as.factor(letter)~., data = train, gamma = 10^(-6:-1), cost = 10^(1:2))



ldatp <- function(prop, g, c, ker) 
{
  data_split <- initial_split(ldat, prop = prop)
  
  train <- training(data_split)
  test  <- testing(data_split)
  test <- data.frame(test)
  
  x <- svm(as.factor(letter)~., data = train, gamma = g, cost = c, kernel = ker)
  prediction <- predict(x, test[,-1])
  
  tab <- table(pred = prediction, true = test[,1])
  tab  
  
  y <- table(prediction == test$letter)
  return(y[2]/(y[1] + y[2]))
}

ldatp(.75, 0.1, 10, 'linear')

ldatp(.75, 0.1, 10, 'polynomial')

ldatp(.75, 0.1, 10, 'radial')

ldatp(.75, 0.1, 10, 'sigmoid')

ldatp(.75, 0.01, 10, 'linear')

ldatp(.75, 0.01, 10, 'polynomial')

ldatp(.75, 0.01, 10, 'radial')

ldatp(.75, 0.01, 10, 'sigmoid')

ldatp(.75, 0.1, 5, 'linear')

ldatp(.75, 0.1, 5, 'polynomial')

ldatp(.75, 0.1, 5, 'radial')

ldatp(.75, 0.1, 5, 'sigmoid')

ldatp(.75, 0.01, 5, 'linear')

ldatp(.75, 0.01, 5, 'polynomial')

ldatp(.75, 0.01, 5, 'radial')

ldatp(.75, 0.01, 20, 'sigmoid')








