library(tidyverse)

#Taking a look at the data
str(train)
str(test)
table(train$level)
range(train$date)
range(test$date)

#Create is_swing variable in train set
table(train$pitch_call)
factor(train$is_swing, levels = c(1,0))

`%notin%` <- Negate(`%in%`)

swings <- c("FoulBall", "InPlay","StrikeSwinging")

train <- mutate(train, is_swing = as.factor(case_when(pitch_call %in% swings ~ 1,
                                                      pitch_call %notin% swings ~ 0)))

#Select variables to include in data, these are the things I think will most impact swing decision:
# level, Pitcher Handedness, Hitter Handedness, Balls, Strikes, Outs, 
# Release Speed, Induced VBreak, HBreak, plate height, plate side, pitch type
# 

train2 <- train[,c(2,4,6,12:15, 25:28, 35, 38)]

##Looking at distribution of numeric variables
summary(train2$release_speed)
summary(train2$induced_vert_break)
summary(train2$horz_break)
summary(train2$plate_height)
summary(train2$plate_side)

#Removing improbable values/misreads (ex. -45.5 and +69.9 induced vertical break is probably not accurate)

# Because this is to train a classification model and there is such a large training set,
# we can be ok with removing misread-like values entirely

train2 <- filter(train2, abs(induced_vert_break) < 30) # +/- 30 inches of induced VBreak is above even the best pitches
train2 <- filter(train2, abs(horz_break) < 30)   #Same for HBreak

table(train2$outs)
table(train2$balls)
table(train2$strikes)
table(train2$pitcher_side)
table(train2$batter_side)

# We also see some observations with 3 strikes, 4 balls, 3 outs, etc. These can't be right and should be removed
train2 <- filter(train2, outs < 3)
train2 <- filter(train2, strikes <3)
train2 <- filter(train2, balls < 4)

#Making a temporary smaller data frame to test model fits
train3 <- na.exclude(train2)

temp <- train3[sample(nrow(train3), 10000),]

library(caret)

#Partitioning the smaller data frame into test and train
set.seed(1)
new=createDataPartition(y=temp$is_swing,p=.7,list=FALSE)
temp_train=temp[new,]
temp_test=temp[-new,]

#Start testing a few model options to see which performs best
tc <- trainControl(method ="cv",10)

# Logistic regression
fit.glm <- train(is_swing~level + pitcher_side + batter_side + outs + balls + strikes + release_speed +
                   induced_vert_break + horz_break + plate_height + plate_side + pitch_type, 
                 data = temp_train, method = "glm", trControl = tc)

preds1 <- predict(fit.glm, temp_test)
confusionMatrix(preds1, temp_test$is_swing) #Logit model has 59.89% accuracy

# Gradient Boosted Machine
grid=expand.grid(n.trees=c(100,200,500),
                 shrinkage=c(0.01,0.05,0.1),
                 n.minobsinnode = c(5),
                 interaction.depth=c(1:3))

fit.gbm <- train(is_swing~level + pitcher_side + batter_side + outs + balls + strikes + release_speed +
                   induced_vert_break + horz_break + plate_height + plate_side + pitch_type, 
                 data = temp_train, method = "gbm", bag.fraction=0.5,
                 verbose = FALSE,
                 tuneGrid=grid, trControl = tc)

preds2 <- predict(fit.gbm, temp_test)
confusionMatrix(preds2, temp_test$is_swing) # GBM - 76.36% accuracy

#Neural Network
library(nnet)
fit.nnet <- nnet(is_swing ~ level + pitcher_side + batter_side + outs + balls + strikes + release_speed +
                   induced_vert_break + horz_break + plate_height + plate_side + pitch_type, 
                 data= temp_train, 
                 method="nnet", 
                 size = 5, 
                 decay = .0001,
                 maxit = 500)

preds3 <- predict(fit.nnet, temp_test)
preds3 <- round(preds3, 0)
preds3 <- as.factor(preds3)
confusionMatrix(preds3,temp_test$is_swing) # Neural Net - 69.49% accuracy

#Random Forrest
library(randomForest)
fit.rf <- randomForest(is_swing~level + pitcher_side + batter_side + outs + balls + strikes + release_speed +
                         induced_vert_break + horz_break + plate_height + plate_side + pitch_type, 
                       data = temp_train, ntree = 1500, random_state = 0)

preds4 <-predict(fit.rf,temp_test)
confusionMatrix(preds4,temp_test$is_swing) # Random Forest - 76.39% accuracy


## GBM and RF are the two best models - look closely at both to decide which to use
# ROC Curves:
library(ROSE)
roc.curve(temp_test$is_swing, preds2) #GBM - .762 AOC
roc.curve(temp_test$is_swing, preds4) #RF - .761 AOC

#Preliminary results show pretty similar performance for GBM and RF, I'm going to choose the RF for ease of 
#  tuning and to reduce risk of overfitting

#Now, run a couple more preliminary models, changing tuning parameters to create final model
# plot first RF
plot(fit.rf) # we see errors converge at around 1500 trees, we'll leave ntree at 1500
tuneRF(temp_train[,1:12], temp_train[,13], stepFactor = 1) # this will give us optimal value for mtry
rf2 <- randomForest(is_swing~level + pitcher_side + batter_side + outs + balls + strikes + release_speed +
                      induced_vert_break + horz_break + plate_height + plate_side + pitch_type, 
                    data = temp_train, ntree = 1500, mtry = 3, random_state = 0)
preds5 <-predict(rf2,temp_test)
confusionMatrix(preds5,temp_test$is_swing)
plot(rf2) 

#Now to fit a RF with these parameters to larger train set

set.seed(1)
train4 <- train3[sample(nrow(train3), 50000),]

model <- randomForest(is_swing~level + pitcher_side + batter_side + outs + balls + strikes + release_speed +
                        induced_vert_break + horz_break + plate_height + plate_side + pitch_type, 
                      data = train4, ntree = 1500, mtry = 3, random_state = 0)

#predict is_swing in test set
test$is_swing <- predict(model, test)

#Look at tables of is_swing between train and test for reasonable results
table(train$is_swing)
table(test$is_swing)  #Distributions look similar, model results are reasonable

