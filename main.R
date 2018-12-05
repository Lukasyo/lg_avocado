# Import, clean & transform data --------------------------------------------------------

library(Matrix)
library(dplyr)
library(xgboost)
library(RcppRoll)
library(reshape2)
library(MLmetrics)
library(ggplot2)

# import data
df <- read.csv('avocado.csv')

# model only conventional avocado price
df <- df[df$type == "conventional",]

# create ids
df$id <- 1:nrow(df)

# convert to date format
df$Date <- as.Date(df$Date)


# Prepare data for modelling ----------------------------------------------

# inspect structure and main statistics
str(df)
summary(df) # no serious problems observerd

#scale features
df[,c("Total.Volume","X4046","X4225","X4770","Total.Bags","Small.Bags","Large.Bags",
      "XLarge.Bags")] <- scale(df[,c("Total.Volume","X4046","X4225","X4770","Total.Bags",
                                     "Small.Bags","Large.Bags","XLarge.Bags")] )

# transform dependent variable to log
df$pricelog <- log1p(df$AveragePrice)

# separate training data
train <- df[df$Date < "2018-02-04",]

#create lags
train <- train[order(train$Date),]

train <- train %>%
  group_by(region) %>%
  mutate(lag_1 = lag(pricelog, 1)
         , avg_8 = lag(roll_meanr(pricelog, 8), 1)
         , avg_3 = lag(roll_meanr(pricelog, 3), 1)
  )

train <- train[!is.na(train$avg_8),] #remove NAs

#Build matrix input for the model
trainMatrix <- sparse.model.matrix(~ avg_3 + avg_8 + lag_1 + Total.Volume + X4046 + X4225 + X4770 + Total.Bags + Small.Bags +
                                     Large.Bags + XLarge.Bags
                                   , data = train
                                   , sparse = FALSE, sci = FALSE)


# Modeling ----------------------------------------------------------------

#Create input for xgboost
label <- train$pricelog
trainDMatrix <- xgb.DMatrix(data = trainMatrix, label = label)

#Set parameters of model
params <- list(booster = "gbtree"
               , objective = "reg:linear"
               , eta=0.4
               , gamma=0)

#Cross-validation
xgb.tab <- xgb.cv(data = trainDMatrix
                  , param = params
                  , maximize = FALSE, evaluation = "rmse", nrounds = 100
                  , nthreads = 10, nfold = 2, early_stopping_round = 10)

# number of rounds
num_iterations <-  xgb.tab$best_iteration

# model
model <- xgb.train(data = trainDMatrix
                   , param = params
                   , maximize = FALSE, evaluation = 'rmse', nrounds = num_iterations)

# feature importance
importance <- xgb.importance(feature_names = colnames(trainMatrix), model = model)
xgb.ggplot.importance(importance_matrix = importance)


# Prediction --------------------------------------------------------------

test <- df

#Dates vector for the testing dataset
Dates <- sort(unique(df[df$Date >= "2018-02-04","Date"]))

#Order test dataset
test <- test[order(test$Date),]

#Create lag variables on the testing data
test <- test %>%
  group_by(region) %>%
  mutate(lag_1 = lag(pricelog, 1)
         , avg_8 = lag(roll_meanr(pricelog, 8), 1)
         , avg_3 = lag(roll_meanr(pricelog, 3), 1)
  )

# prediction (a loop is required because as we use lags, we only have them for the 1st prediction period, for the further
# ones we would need to use the predictions made for calculating lags and moving averages)

for (i in 1:length(Dates)){
  
  #Subset testing data to predict only 1 week at a time
  Stest <- test[test$Date == as.character(Dates[i]),]
  
  #Remove NAs
  Stest <- Stest[!is.na(Stest$avg_8),]

  
  testMatrix <- sparse.model.matrix(~ avg_3 + avg_8 + lag_1 + Total.Volume + X4046 + X4225 + X4770 + Total.Bags + Small.Bags +
                                                     Large.Bags + XLarge.Bags
                                                   , data = Stest
                                                   , sparse = FALSE, sci = FALSE)
  
  
  #Predict values for a given week
  pred <- predict(model, testMatrix)
  
  #Add predicted values to the data set based on ids
  test$pricelog_pred[test$id %in% Stest$id] <- pred
  
  print(i)
  gc()
  
}


# Post-modeling clean-up --------------------------------------------------

#Convert predictions back to normal unit from the log.
test$AveragePricePred <- expm1(test$pricelog_pred)

#add to main dataframe
df <- left_join(df,test[,c("id","AveragePricePred")])

#join historical data
df[is.na(df$AveragePricePred),"AveragePricePred"] <- df[is.na(df$AveragePricePred),"AveragePrice"]


# Results overview --------------------------------------------------------

# example graph
graph_data <- df[df$region == "Detroit" & df$Date > "2017-06-01",c("AveragePrice","AveragePricePred","Date")]
graph_data <- melt(graph_data,id="Date")

ggplot(data=graph_data, aes(x=Date, y=value, group=variable)) +
  geom_line(aes(color=variable)) +
  ggtitle("Avocado price in Detroit")

# MAPE
MAPE(df[df$Date >= "2018-02-04","AveragePricePred"],df[df$Date >= "2018-02-04","AveragePrice"])
