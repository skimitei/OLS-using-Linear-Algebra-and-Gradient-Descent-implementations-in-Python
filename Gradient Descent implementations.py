#  By Symon Kimitei
#  Gradient Descent Implementation 
#-----------------------------------------------------------
# initialize the weights
b = np.zeros(shape=len(train_X.columns)) 

# this will hold the calculated cost at each epoch(iteration)
cost_history = [] 

# calculate the partial derivative with respect to b_i and the cost in each epoch(iteration)
for i in range(1000):
    # calculates y hat (train)
    train_yhat = train_X.dot(b) 

    # calculate the partial derivative with respect to b_i
    Db = (-2/len(train_X))*((train_y-train_yhat).dot(train_X)) 
    
    # small step size to prevent coefficients from increasing to positive infinity
    b-= 0.0000001*Db 
    
    current_cost = cost(train_yhat,train_y)
    # Save the cost history at each iteration
    cost_history.append(current_cost)
    
    # display iteration number and cost at each iteration.
    print("iteration",str(i) + ":  Cost =",current_cost) 
b_opt=b


# In[8]:


# Display a graph of Cost Vs Epoch
plt.plot(cost_history)
plt.title("Calculated Cost by Epoch")
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.show()


# In[14]:


# prints out the coefficients:
print("Coefficients:")
for (pixel,coef) in dict(zip(train_X.columns,b_opt)).items():
    print(pixel + ":",coef)

# classifying all predicted values above .5 as 1 and below .5 as zero (train)
train_yclass = (train_X.dot(b_opt) > .5).values 
print("Training Classification Accuracy:",np.mean(train_yclass == train_y))

# classifying all predicted values above .5 as 1 and below .5 as zero (test)
test_yclass = (test_X.dot(b_opt) > .5).values
print("Test Classification Accuracy:",np.mean(test_yclass == test_y))

