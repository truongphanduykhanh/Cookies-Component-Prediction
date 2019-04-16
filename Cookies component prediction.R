#                                   Khanh TRUONG
#                                   April 2019
#                           Cookie component prediction
# ------------------------------------------------------------------------------

library(ggplot2) # for plotting
library(gridExtra) # for arrange plot
library(ppls) # for coockie data
library(DDoutlier) # for outliner detection (LOF function)
library(glmnet) # for ridge regression
library(pls) # for PLS regression
library(lars) # lasso regression
library(neuralnet) # for neural network

theme_update(plot.title = element_text(hjust = 0.5)) #adjust the tittle of plots


# Prepare data -----------------------------------------------------------------
data(cookie)

# Extraction the sugar rate and spectrum
cookie_sugar = data.frame(cookie[,702],cookie[,1:700])
names(cookie_sugar)= c("sugar",paste("X",1:700,sep=""))

# Split train set and test set
train = cookie_sugar[1:40,]
test = cookie_sugar[41:nrow(cookie_sugar),]


# Outlier detection ------------------------------------------------------------
LOF_scores_1 = LOF(train, k=3) # LOF score when k=3
LOF_scores_2 = LOF(train, k=5) # LOF score when k=5
LOF_scores_3 = LOF(train, k=7) # LOF score when k=7

LOF1 <- ggplot() +
  geom_point(aes(x=seq(1:nrow(train)), y=LOF_scores_1)) +
  labs(x=NULL, y='LOF score', title='K=3')
  
LOF2 <- ggplot() +
  geom_point(aes(x=seq(1:nrow(train)), y=LOF_scores_2)) +
  labs(x=NULL, y='LOF score', title='K=5')

LOF3 <- ggplot() +
  geom_point(aes(x=seq(1:nrow(train)), y=LOF_scores_3)) +
  labs(x='Observation', y='LOF score', title='K=7')

grid.arrange(LOF1, LOF2, LOF3)



# remove the outliner
train = train[-which.max(LOF_scores_1), ]
rm(LOF_scores_1, LOF_scores_2, LOF_scores_3,
   LOF1, LOF2, LOF3)

# Standardize data for models --------------------------------------------------
train[names(train)!='sugar'] = scale(train[names(train)!='sugar'],
                                     center = TRUE,
                                     scale = TRUE)

test[names(test)!='sugar'] = scale(test[names(test)!='sugar'],
                                     center = TRUE,
                                     scale = TRUE)


# Ridge regression -------------------------------------------------------------

# Ridge regression with several lambda
ridge_initial = lm.ridge(sugar~., data = train, lambda=seq(0, 1, 0.01))
plot(ridge_initial)


# Use cross validation to choose best lambda
set.seed(1234)
cv.ridge = cv.glmnet(x = as.matrix(train[names(train)!='sugar']),
                     y = as.matrix(train['sugar']),
                     alpha = 0,
                     lambda = seq(0, 1, 0.01),
                     type.measure='mse') # Use Mean-Squared-Error as loss function
plot(cv.ridge) # MSE versus log(lambda) - choose the lowest log(lambda)

(opt_lambda <- cv.ridge$lambda.min) # best lambda
ridge <- cv.ridge$glmnet.fit # model with best lambda

# Make prediction
pred_train_ridge = predict(ridge,
                           s = opt_lambda,
                           newx = as.matrix(train[names(train)!='sugar']))

pred_test_ridge = predict(ridge,
                          s = opt_lambda,
                          newx = as.matrix(test[names(test)!='sugar']))

# Evaluate model

## on train set
(sst_train_ridge <- sum((train$sugar - mean(train$sugar))^2)) # Sum of Squares Total
(sse_train_ridge <- sum((pred_train_ridge - train$sugar)^2)) # Sum of Squares Error
(rsq_train_ridge <- 1 - sse_train_ridge / sst_train_ridge) # R squared
(rmse_train_ridge <- (sse_train_ridge / nrow(train))^(1/2)) # Root Mean Square Error

## on test set
(sst_test_ridge <- sum((test$sugar - mean(test$sugar))^2)) # Sum of Squares Total
(sse_test_ridge <- sum((pred_test_ridge - test$sugar)^2)) # Sum of Squares Error
(rsq_test_ridge <- 1 - sse_test_ridge / sst_test_ridge) # R squared
(rmse_test_ridge <- (sse_test_ridge / nrow(test))^(1/2)) # Root Mean Square Error

rm(ridge_initial, cv.ridge, opt_lambda, ridge,
   pred_train_ridge, pred_test_ridge)


# PLS regression ---------------------------------------------------------------
set.seed(1234)
plsFit = plsr(sugar ~ ., data=train, validation="CV")
summary(plsFit)

# Find optimal number of components
validationplot(plsFit, val.type="MSEP")
abline(v = 5, lty=2)

validationplot(plsFit, val.type="R2")
abline(v = 5, lty=2)

optimal_comp = 5


# Make prediction
pred_train_pls = predict(plsFit,
                         as.matrix(train[names(train)!='sugar']),
                         ncomp=optimal_comp) # on train set

pred_test_pls = predict(plsFit,
                        as.matrix(test[names(test)!='sugar']),
                        ncomp=optimal_comp) # on test set

# Evaluate model

## on train set
(sst_train_pls <- sum((train$sugar - mean(train$sugar))^2)) # Sum of Squares Total
(sse_train_pls <- sum((pred_train_pls - train$sugar)^2)) # Sum of Squares Error
(rsq_train_pls <- 1 - sse_train_pls / sst_train_pls) # R squared
(rmse_train_pls <- (sse_train_pls / nrow(train))^(1/2)) # Root Mean Square Error

## on test set
(sst_test_pls <- sum((test$sugar - mean(test$sugar))^2)) # Sum of Squares Total
(sse_test_pls <- sum((pred_test_pls - test$sugar)^2)) # Sum of Squares Error
(rsq_test_pls <- 1 - sse_test_pls / sst_test_pls) # R squared
(rmse_test_pls <- (sse_test_pls / nrow(test))^(1/2)) # Root Mean Square Error

rm(plsFit, optimal_comp, pred_train_pls, pred_test_pls)


# Lasso estimator --------------------------------------------------------------
lasso = lars(x = as.matrix(train[names(train)!='sugar']),
             y = as.matrix(train['sugar']),
             type = 'lar',
             use.Gram=FALSE)
plot(lasso)


# estimate optimal position in path
set.seed(1234)
cv.lasso = cv.lars(x = as.matrix(train[names(train)!='sugar']),
                   y = as.matrix(train['sugar']),
                   type = 'lar',
                   use.Gram=FALSE)

s.cv <- which.min(cv.lasso$cv) # choose the best number of step
coef(lasso, s=s.cv) # coefficients
sum(coef(lasso, s=s.cv)!=0) # number of non-zero coefficients


# Make prediction
pred_train_lasso = predict(lasso,
                           as.matrix(train[names(train)!='sugar']),
                           s=s.cv)$fit # on train set

pred_test_lasso = predict(lasso,
                          as.matrix(test[names(test)!='sugar']),
                          s=s.cv)$fit # on test set

# Evaluate model

## on train set
(sst_train_lasso <- sum((train$sugar - mean(train$sugar))^2)) # Sum of Squares Total
(sse_train_lasso <- sum((pred_train_lasso - train$sugar)^2)) # Sum of Squares Error
(rsq_train_lasso <- 1 - sse_train_lasso / sst_train_lasso) # R squared
(rmse_train_lasso <- (sse_train_lasso / nrow(train))^(1/2)) # Root Mean Square Error

## on test set
(sst_test_lasso <- sum((test$sugar - mean(test$sugar))^2)) # Sum of Squares Total
(sse_test_lasso <- sum((pred_test_lasso - test$sugar)^2)) # Sum of Squares Error
(rsq_test_lasso <- 1 - sse_test_lasso / sst_test_lasso) # R squared
(rmse_test_lasso <- (sse_test_lasso / nrow(test))^(1/2)) # Root Mean Square Error

rm(lasso, cv.lasso, s.cv,
   pred_train_lasso, pred_test_lasso)


# Neural network ---------------------------------------------------------------

# Choose number of neurons in each layers
neurons_1 = as.integer(ncol(train)*1/3) # first hiden: one third of input layer
neurons_2 = as.integer(neurons_1*1/3) # second hiden: one third of first hidden

set.seed(0)
nn <- neuralnet(sugar~.,
                data=train,
                hidden=c(neurons_1, neurons_2),
                learningrate=0.01,
                linear.output=TRUE)


# Make prediction
pred_train_nn = compute(nn, train[names(train)!='sugar'])$net.result # train set
pred_test_nn = compute(nn, test[names(test)!='sugar'])$net.result # on test set


# Evaluate model

## on train set
(sst_train_nn <- sum((train$sugar - mean(train$sugar))^2)) # Sum of Squares Total
(sse_train_nn <- sum((pred_train_nn - train$sugar)^2)) # Sum of Squares Error
(rsq_train_nn <- 1 - sse_train_nn / sst_train_nn) # R squared
(rmse_train_nn <- (sse_train_nn / nrow(train))^(1/2)) # Root Mean Square Error

## on test set
(sst_test_nn <- sum((test$sugar - mean(test$sugar))^2)) # Sum of Squares Total
(sse_test_nn <- sum((pred_test_nn - test$sugar)^2)) # Sum of Squares Error
(rsq_test_nn <- 1 - sse_test_nn / sst_test_nn) # R squared
(rmse_test_nn <- (sse_test_nn / nrow(test))^(1/2)) # Root Mean Square Error

rm(nn, pred_train_nn, pred_test_nn)