
import numpy as np
from matplotlib import pyplot as plt
from jupyterthemes import jtplot
jtplot.style()


meanApple = np.array([7.0,8.0]) ## 2 features mean value of x(color) is 7.0 and sweetness is 8.0

# x - color and y - sweetness
## covariance matrix - Sigma(xy) means how changing x(color) will have effect on y(sweetness)
## so more red apples means they are more sweet so 0.5 deviation(sigma)

covApple = np.array([[1.0,-0.5],[-0.5,1.0]])  ## 2 X 2 matrix


distributionApple = np.random.multivariate_normal(meanApple,covApple,2000)  # 2000 apple pts
# HERE 20000 means no of points or rows. Cols are
# no of features or size of mean array i.e 2 here in this case


print distributionApple.shape  # 2000 X 2 tuple

meanLemon = np.array([3.0,4.0])

covLemon = np.array([[1.0,0.5],[0.5,1.0]])

distributionLemon = np.random.multivariate_normal(meanLemon,covLemon,2000)	 # 2000 lemon pts

plt.figure(0)

plt.scatter(distributionApple[:,0],distributionApple[:,1],color ='red')
plt.scatter(distributionLemon[:,0],distributionLemon[:,1],color = 'yellow')
plt.xlabel('Color')
plt.ylabel('Sweetness')
plt.show()


# Training and Testing Data Preparation

# 3000 Samples - 1500 Apples, 1500 for Lemons


labelsTrain = np.zeros((3000,1))
labelsTrain[:1500] = 1.0

trainingArr = np.zeros((3000,2))
trainingArr[:1500,:] = distributionApple[:1500,:]  ## first 1500 rows are for apple points
trainingArr[1500:,:] = distributionLemon[:1500,:] ## from 1500 rows are for lemon data

labelsTest = np.zeros((1000,1))
labelsTest[:500] = 1.0

testingArr = np.zeros((1000,2))
testingArr[:500,:] = distributionApple[1500:2000,:]
testingArr[500:,:] = distributionLemon[1500:2000,:]

print labelsTrain
print trainingArr
print labelsTest
print testingArr


# KNN Algorithm :)
# Dist of the query_point to all other points in the space ( O(N)) time for every point + sorting
# the complexity is O(Q.N)


#Euclidean Distance
def dist(p1,p2):
	return np.sqrt(((p2-p1)**2).sum())


# value of k should always be odd
def knn(trainA,labelA,query_point, k) :

	val = []

	for i in range(trainA.shape[0]) :
		v = [ dist(query_point,trainA[i,:]), labelA[i] ]
		val.append(v)

	#sort the array acc to distances
	val = sorted(val)

	# pick top k nearest distances
	predArr = np.array(val[:k])

	# find frequency of each label and print the label whose frequency is more
	newPredArr = np.unique(predArr[:,1],return_counts = True)

	index = newPredArr[1].argmax()

	return newPredArr[0][index]



# calculate value of k i.e. square root of number of samples in your training dataset
import math
k = math.floor(math.sqrt(trainingArr.shape[0]))
k = int(k)
if k%2 == 0 :
	k += 1

print k



arr = []
for i in range(testingArr.shape[0]) :
	arr.append(knn(trainingArr,labelsTrain,testingArr[i,:],k))

np_arrPred = np.array(arr)
# so your np_arrPred contains 1st 500 ones and next 500 will be zeros according to labelsTest
print np_arrPred


# Now find the accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(np_arrPred,labelsTest)
print accuracy


# We can find accuracy manually also like this
count = 0.0

size = testingArr.shape[0] # size is 1000

for i in range(size) :
	if i < size/2 :
		if np_arrPred[i] == 1 :
			count += 1
	else :
		if np_arrPred[i] == 0 :
			count += 1


my_accuracy = (count/size)

print("my accuracy is : %f" %my_accuracy)


###### in both the cases accuracy is approx 98.8% which is correct



from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(trainingArr, np.ravel(labelsTrain))
print "accuracy of knn algorithm is : ", clf.score(testingArr,labelsTest)

# accuracy of sklearn mathches with our knn algorithm's accuarcy
