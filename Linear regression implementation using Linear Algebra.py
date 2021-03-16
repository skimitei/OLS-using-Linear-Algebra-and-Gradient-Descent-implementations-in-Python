#  By Symon Kimitei
#  Linear regression implementation using Linear Algebra
#----------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#==========================================================================
# Download the dataset
os.chdir("C:/Users/kimit/OneDrive/Desktop/py/hw2")

# read in the train data set into pandas dataframes and add a constant 
# as a predictor in the training data set 
train_df = pd.read_csv("lr_training.csv") # x
train_X = train_df.drop("label",axis=1)
train_X["Constant"] = 1 # constant for b0
train_y = train_df.label # y

train_X.head(5)


# In[2]:


# read in the test data set into pandas dataframes and add a constant 
# as a predictor in the test data set 
test_df = pd.read_csv('lr_test.csv')
test_X = test_df.drop("label",axis=1) # x
test_y = test_df.label # y
test_X['Constant'] = 1 # constant b0

test_X.head(5)


# In[3]:


### Solving by using the OLS algorithm ###
#----------------------------------------------------------
# Derive the beta values utilizing the closed form solution

# the calculation of the optimal beta weights using linear algebra
b_opt = np.linalg.pinv(train_X.T.dot(train_X)).dot(train_X.T).dot(train_y)
b_opt


# In[10]:


# using the beta weights and data to predict y hat (train) 
train_yhat = train_X.dot(b_opt) 

# the cost function
def cost(yhat,y):
    return np.mean((yhat - train_y)**2)

# calculates the cost function for X train
print(cost(train_yhat,train_y)) 


# In[11]:


# using the beta weights and data to predict y hat (test)
test_yhat = test_X.dot(b_opt) 

# calculates the cost function for X test 
print(cost(test_yhat,test_y)) 


# In[12]:


# classifying all predicted values above .5 as 1 and below .5 as 0 (train)
train_yclass = (train_X.dot(b_opt) > .5).values 

# prints out the coefficients:
print("Coefficients:")
for (pixel,coef) in dict(zip(train_X.columns,b_opt)).items():
    print(pixel + ":",coef)

# finds the proportion of correctly predicted classes (train)
print("Training Classification Accuracy:",np.mean(train_yclass == train_y)) 

# classifying all predicted values above .5 as 1 and below .5 as 0 (test)
test_yclass = (test_X.dot(b_opt) > .5).values

# finds the proportion of correctly predicted classes (train)
print("Test Classification Accuracy:",np.mean(test_yclass == test_y)) 

