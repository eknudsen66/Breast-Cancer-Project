#Using the brca dataset from the dslabs package contains informatuon about breast cancer diagnosis biopsy samples for tumors determined to be malignant or benign
#Load data and libraries
options(digits = 3)
library(matrixStats)
library(tidyverse)
library(caret)
library(dslabs)
data(brca)

#Number of samples within dataset
dim(brca$x)[1]
569

#Proportion of malignant samples
mean(brca$y == "M")
0.373

#Scaling matrix
x_centered <- sweep(brca$x, 2, colMeans(brca$x))
x_scaled <- sweep(x_centered, 2, colSds(brca$x), FUN = "/")

sd(x_scaled[,1])

#Produce heat map of relationship between features using the scaled matrix
d_features <- dist(t(x_scaled))
heatmap(as.matrix(d_features), labRow = NA, labCol = NA)

#Principal component analysis of scaled matrix
pca <- prcomp(x_scaled)
summary(pca)

#Plot two principal components with color representing tumor type (M,B)
data.frame(pca$x[,1:2], type = brca$y) %>%
  ggplot(aes(PC1, PC2, color = type)) +
  geom_point()

#Training and test sets
set.seed(1)
test_index <- createDataPartition(brca$y, times = 1, p = 0.2, list = FALSE)
test_x <- x_scaled[test_index,]
test_y <- brca$y[test_index]
train_x <- x_scaled[-test_index,]
train_y <- brca$y[-test_index]

#Proportion of training set benign
mean(train_y == "B")
0.628
#Test set
mean(test_y == "B")
0.626

#Using predict_kmeans() to make predictions on test set
predict_kmeans <- function(x, k) {
  centers <- k$centers
  distances <- sapply(1:nrow(x), function(i){
    apply(centers, 1, function(y) dist(rbind(x[i,], y)))
  })
  max.col(-t(distances))  # select cluster with min distance to center
}
#Overall accuracy
set.seed(3)
k <- kmeans(train_x, centers = 2)
kmeans_preds <- ifelse(predict_kmeans(test_x, k) == 1, "B", "M")
mean(kmeans_preds == test_y)
0.896

#Proportion of benign tumors correctly identified
sensitivity(factor(kmeans_preds), test_y, positive = "B")
0.958

#Proportion of malignant tumors correctly identified
sensitivity(factor(kmeans_preds), test_y, positive = "M")
0.791

#Testing accuracy of logistic regression model
train_glm <- train(train_x, train_y, method = "glm")
glm_preds <- predict(train_glm, test_x)
mean(glm_preds == test_y)
0.939

#loess model accuracy
set.seed(5)
train_loess <- train(train_x, train_y, method = "gamLoess")
loess_preds <- predict(train_loess, test_x)
mean(loess_preds == test_y)
0.93

#K-nearest neighbors model
set.seed(7)
tuning <- data.frame(k = seq(3, 21, 2))
train_knn <- train(train_x, train_y,
                   method = "knn", 
                   tuneGrid = tuning)
train_knn$bestTune
knn_preds <- predict(train_knn, test_x)
mean(knn_preds == test_y)
0.974

#Random forest model accuracy
set.seed(9)
tuning <- data.frame(mtry = c(3, 5, 7, 9))
train_rf <- train(train_x, train_y,
                  method = "rf",
                  tuneGrid = tuning,
                  importance = TRUE)
train_rf$bestTune
rf_preds <- predict(train_rf, test_x)
mean(rf_preds == test_y)
0.948

#LDA model accuracy
train_lda <- train(train_x, train_y, method = "lda")
lda_preds <- predict(train_lda, test_x)
mean(lda_preds == test_y)
0.974

#QDA model accuracy
train_qda <- train(train_x, train_y, method = "qda")
qda_preds <- predict(train_qda, test_x)
mean(qda_preds == test_y)
0.948

#Create ensemble to generate majority prediction of tumor type
ensemble <- cbind(glm = glm_preds == "B", lda = lda_preds == "B", qda = qda_preds == "B", loess = loess_preds == "B", rf = rf_preds == "B", knn = knn_preds == "B", kmeans = kmeans_preds == "B")

ensemble_preds <- ifelse(rowMeans(ensemble) > 0.5, "B", "M")
mean(ensemble_preds == test_y)
0.961

#Calculate most accurate model
models <- c("K means", "Logistic regression", "LDA", "QDA", "Loess", "K nearest neighbors", "Random forest", "Ensemble")
accuracy <- c(mean(kmeans_preds == test_y),
              mean(glm_preds == test_y),
              mean(lda_preds == test_y),
              mean(qda_preds == test_y),
              mean(loess_preds == test_y),
              mean(knn_preds == test_y),
              mean(rf_preds == test_y),
              mean(ensemble_preds == test_y))
data.frame(Model = models, Accuracy = accuracy)
LDA