---
title: 'Peer-graded Assignment: Regression Models Course Project'
output:
  html_document:
    keep_md: yes
  pdf_document: default
---



## Overview

The goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which they did the exercise.

### Setting up

1. Download all the files and install all the relevant packages.


```r
set.seed(123)
urltrain<-'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv'
urltest<-'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'
training<-read.csv(urltrain,na.strings=c('','NA'))
testing<-read.csv(urltest,na.strings=c('','NA'))
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(dplyr)
```

```
## 
## Attaching package: 'dplyr'
```

```
## The following objects are masked from 'package:stats':
## 
##     filter, lag
```

```
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

```r
library(rattle)
```

```
## Rattle: A free graphical interface for data science with R.
## Version 5.3.0 Copyright (c) 2006-2018 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

2. Exploratory data analysis


```r
dim(training);dim(testing)
```

```
## [1] 19622   160
```

```
## [1]  20 160
```

```r
qplot(classe,data=training)
```

![](Figs/unnamed-chunk-2-1.png)<!-- -->

```r
newtraining<-training[,colSums(is.na(training))==0] ## remove columns/variables that have NA
newtesting<-testing[,colSums(is.na(testing))==0] ## remove columns/variables that have NA
dim(newtraining);dim(newtesting)
```

```
## [1] 19622    60
```

```
## [1] 20 60
```


```r
length(intersect(colnames(newtraining),colnames(newtesting))) ## find out how many common variables are there
```

```
## [1] 59
```


```r
setdiff(colnames(newtraining),intersect(colnames(newtraining),colnames(newtesting))) ## find out which is the variable that does not overlap
```

```
## [1] "classe"
```
3. Cross Validation

This is done by using the Validation set approach.


```r
intrain<-createDataPartition(y=newtraining$classe,p=0.7,list=FALSE)
training1<-newtraining[intrain,]
validation<-newtraining[-intrain,]
```

### Methods

1. Linear Discriminant Analysis (Model-based prediction)

The LDA method is known to be computationally fast but lacks accuracy.


```r
modellda<-train(classe~.,method='lda',data=training1)
```

```
## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear
```

```r
ldapredict<-predict(modellda,validation)
confusionMatrix(ldapredict,validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    0 1139    0    0    0
##          C    0    0 1026    0    0
##          D    0    0    0  964    0
##          E    0    0    0    0 1082
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9994, 1)
##     No Information Rate : 0.2845     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##                                      
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

A suprising observation here is that the accuracy is unusually high. Hence, it is worth to further investigate what are the key variables involved in the outcome prediction.


```r
varimpdata<-varImp(modellda) 
head(varimpdata$importance,5)
```

```
##                                A          B          C          D          E
## X                    100.0000000 100.000000 100.000000 100.000000 100.000000
## user_name              4.2087064   4.208706   4.208706   4.572903   4.124892
## raw_timestamp_part_1  24.9192075  24.919208  24.919208  24.919208  20.382845
## raw_timestamp_part_2   0.8655634   1.014740   1.441471   1.016279   1.014740
## cvtd_timestamp        12.6825779  17.223724  11.646927  12.162809  17.223724
```

It seems that variable X is able to perfectly predict the outcome, hence it could be proxy variable of the outcome. The variable should be removed from the training, validation and testing sets. 


```r
training1<-select(training1,-X)
validation<-select(validation,-X)
newtesting<-select(newtesting,-X)
```

Now, we run the LDA method again after removing variable X.


```r
modellda2<-train(classe~.,method='lda',data=training1)
```

```
## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear

## Warning in lda.default(x, grouping, ...): variables are collinear
```

```r
ldapredict2<-predict(modellda2,validation)
confusionMatrix(ldapredict2,validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1521  127    3    0    0
##          B  129  857  109    2    0
##          C   24  150  894   99    7
##          D    0    5   20  809  108
##          E    0    0    0   54  967
## 
## Overall Statistics
##                                           
##                Accuracy : 0.8578          
##                  95% CI : (0.8486, 0.8666)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8202          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9086   0.7524   0.8713   0.8392   0.8937
## Specificity            0.9691   0.9494   0.9424   0.9730   0.9888
## Pos Pred Value         0.9213   0.7812   0.7615   0.8588   0.9471
## Neg Pred Value         0.9639   0.9411   0.9720   0.9686   0.9764
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2585   0.1456   0.1519   0.1375   0.1643
## Detection Prevalence   0.2805   0.1864   0.1995   0.1601   0.1735
## Balanced Accuracy      0.9389   0.8509   0.9069   0.9061   0.9412
```


```r
ldaoutofsample<-1-confusionMatrix(ldapredict2,validation$classe)$overall[1]
ldaoutofsample[[1]]
```

```
## [1] 0.142226
```

As we can see, the accuracy of the model has decreased to an expected level. 

2. Classification Trees

The classification trees method is known to be easy to intepret and better for non-linear relationships.


```r
modeltree<-train(classe~.,method='rpart',data=training1)
treepredict<-predict(modeltree,validation)
confusionMatrix(treepredict,validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1297  326   32   52   12
##          B    4  192    1    0    0
##          C  251  222  873  440  305
##          D  120  399  120  472  278
##          E    2    0    0    0  487
## 
## Overall Statistics
##                                          
##                Accuracy : 0.5643         
##                  95% CI : (0.5515, 0.577)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.4517         
##                                          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.7748  0.16857   0.8509   0.4896  0.45009
## Specificity            0.8998  0.99895   0.7493   0.8137  0.99958
## Pos Pred Value         0.7545  0.97462   0.4175   0.3398  0.99591
## Neg Pred Value         0.9095  0.83351   0.9597   0.8906  0.88973
## Prevalence             0.2845  0.19354   0.1743   0.1638  0.18386
## Detection Rate         0.2204  0.03263   0.1483   0.0802  0.08275
## Detection Prevalence   0.2921  0.03347   0.3553   0.2360  0.08309
## Balanced Accuracy      0.8373  0.58376   0.8001   0.6516  0.72484
```


```r
troutofsample<-1-confusionMatrix(treepredict,validation$classe)$overall[1]
troutofsample[[1]]
```

```
## [1] 0.4356839
```

As we can see, the accuracy of the model here is lower than that of the LDA model, while the estimated out of sample error is higher.


```r
fancyRpartPlot(modeltree$finalModel)
```

![](Figs/unnamed-chunk-13-1.png)<!-- -->

(Explanation)

3. Random Forests

The random forests model is known to be accurate but difficult to intepret and computationally slow.


```r
modelrf<-train(classe~.,method='rf',data=training1)
rfpredict<-predict(modelrf,validation)
confusionMatrix(rfpredict,validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    0 1139    1    0    0
##          C    0    0 1025    0    0
##          D    0    0    0  964    1
##          E    0    0    0    0 1081
## 
## Overall Statistics
##                                      
##                Accuracy : 0.9997     
##                  95% CI : (0.9988, 1)
##     No Information Rate : 0.2845     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 0.9996     
##                                      
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   0.9990   1.0000   0.9991
## Specificity            1.0000   0.9998   1.0000   0.9998   1.0000
## Pos Pred Value         1.0000   0.9991   1.0000   0.9990   1.0000
## Neg Pred Value         1.0000   1.0000   0.9998   1.0000   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1935   0.1742   0.1638   0.1837
## Detection Prevalence   0.2845   0.1937   0.1742   0.1640   0.1837
## Balanced Accuracy      1.0000   0.9999   0.9995   0.9999   0.9995
```


```r
rfoutofsample<-1-confusionMatrix(rfpredict,validation$classe)$overall[1]
rfoutofsample[[1]]
```

```
## [1] 0.0003398471
```

As we can see, the accuracy of the model here is higher than the LDA and classification trees methods, and the estimated out of sample error is also lower.

4. Boosting with Trees

Boosting with trees helps to create a stronger predictor by weighing weak predictors and adding them up. Hence, it is known to be accurate but is computationally slow.


```r
modelbt<-train(classe~.,method='gbm',data=training1,verbose=FALSE)
btpredict<-predict(modelbt,validation)
confusionMatrix(btpredict,validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    0 1137    1    0    0
##          C    0    0 1019    0    0
##          D    0    2    6  961    5
##          E    0    0    0    3 1077
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9971          
##                  95% CI : (0.9954, 0.9983)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9963          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9982   0.9932   0.9969   0.9954
## Specificity            1.0000   0.9998   1.0000   0.9974   0.9994
## Pos Pred Value         1.0000   0.9991   1.0000   0.9867   0.9972
## Neg Pred Value         1.0000   0.9996   0.9986   0.9994   0.9990
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1932   0.1732   0.1633   0.1830
## Detection Prevalence   0.2845   0.1934   0.1732   0.1655   0.1835
## Balanced Accuracy      1.0000   0.9990   0.9966   0.9971   0.9974
```


```r
btoutofsample<-1-confusionMatrix(btpredict,validation$classe)$overall[1]
btoutofsample[[1]]
```

```
## [1] 0.0028887
```

As we can see, the accuracy of the model here is high but lower than that of the random forests model, and the estimated out of sample error is low but higher than that of the random forests model.

### Conclusion

The random forests model (rf) is best in terms of accuracy, hence I decide to use that model to predict on newtesting.


```r
finalpredict<-predict(modelrf,newtesting)
finalpredict
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
