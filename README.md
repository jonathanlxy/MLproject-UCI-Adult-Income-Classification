# MLproject-UCI-Adult-Income-Classification
## A Machine Learning Implementation Project using R and Matlab

Xinyi Liu

## Data Cleaning folder
_R Script for data preparation, along with the original dataset file._

###### DataCleaning.R
An R script for data preparation. By executing this script, it will perform the following tasks:

1. Read data set from "adult.data.txt" file;
	
2. Transform all columns into integers;
	
3. Remove the "education" feature;
	
4. Randomly split dataset into training and testing by 75 : 25 ratio;
	
5. Write training set and testing set into "finalset_cleaned_X.csv" and "finalset_cleaned_Y.csv" perspectively.
	


## SVM folder - Matlab script used for Support Vector Machine Analysis.

###### SVMscript.m
_Matlab script that illustrates the performance of SVM model_


## LR folder - Matlab functions and script used for Logistic Regression Analysis.

###### learnLogisticWeights.m
_Weight learning function that implemented L1 and L2 regualrizations._

**Input Variables:**
- w0
	- Initial weight vector
- x
	- A matrix of features, each row stands for an observation
- y
	- A vector of actual labels
- n
	- Number of iterations this function will run to update w0's value using gradient decent
- re
	- Regularization option. 
		- 0 = no regularization, 
		- 1 = L1 regularization, 
		- 2 = L2 regularization
- lambda
	- Used for updating w0's values when re = 1 or re = 2. The larger lambda will cause less impact on gradient decent process.

- **Output of this function**
	- An updated weight vector.


###### sigmoidLikehood.m
_Takes in features of a single point and a label, return the sigmoid likelihood that the point associated with given label_


###### logisticClassify.m

Take the features of a point, returns the more possible label of this point.
