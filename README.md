# K-nearest-neighbors
<br>

## Introduction:
*The K-nearest-neighbor (KNN) algorithm measures the distance between a query scenario and a set of scenarios in the data set.
KNN falls in the supervised learning family of algorithms. Informally, this means that we are given a labelled dataset consisting of training observations (x,y) and would like to capture the relationship between x and y.
This method used for classification and regression.*

<br>

## Algorithm:
1) Calculate distance of the query point to all other K points in the space. This takes O(N) time for every point.
2) Sort the array and pick 1st k rows of sorted array.
3) Return that label whose count(frequency of label) is more.

If there are Q points then time complexity of algorithm is O(Q.N)

<br>

## Details :
Given a query point identify whether it is an apple or a lemon

Here I have created a 4000 data points for apple and lemon on the fact that: 

| Labels | Red | Yellow | Sweetness | Sourness |
| --- | --- | --- | --- | --- |
| Apples | Higher | Lower | Higher | Lower |
| Lemons | Lower | Higher | Lower | Higher |

<br>

## Scatter plot of these points
<img  src = "https://github.com/codeboy47/K-nearest-neighbors/blob/master/Images/scatterPlot.jpg" />

<br> 

## Accuracy :
I implemented my own K-nearest-neighbors algorithm and compare the accuracy with KNeighborsClassifier. In both the cases accuracy comes out to be 99.6%. 


<img  src = "https://github.com/codeboy47/K-nearest-neighbors/blob/master/Images/accuracy.png" />

<br>
Note : I have used Euclidean Distance for calculating distance between two points.
