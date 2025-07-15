import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.datasets import  make_circles

X, Y =  make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='summer')
Y = Y.reshape((1, Y.shape[0]))          # shape : (1, 100)
X=X.T           # shape : (2, 100)

# debug print(np.shape())

def logloss(A,y):
    return 1/len(y)*np.sum(-y*np.log(A)-(1-y)*np.log(1-A+1e-8))

def initialisation(n1, n0, n2):
    
    m= Y.shape[1]
    W1  = rd.randn(n1 , n0)*(1 / m)**(-1/2)
    b1  = rd.randn(n1 , 1)
    W2  = rd.randn(n2 , n1)*(1 / m)**(-1/2)
    b2  = rd.randn(n2 , 1)
    parameters={'W1':W1,"b1": b1,'W2':W2,"b2": b2}
    return parameters
    
# parameters in a dictionary 

def fp(X, parameters):
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    Z1 = np.dot(W1,X) + b1      # shape : (n1, 100)
    A1=1/(1+np.exp(-Z1))        # shape : (n1, 100)
    Z2 = np.dot(W2,A1) + b2         # shape : (1, 100)
    A2=1/(1+np.exp(-Z2))           # shape : (1, 100)
    
    activation={"A1":A1,"A2":A2}
    
    return activation 

# cout logloss -1/m*sum(Ylog(A)+(1-Y)*log(1-A)))
#  Z = WX+b ou Z= WA1+b

def backpropagation(X,Y,activation,parameters):
    
    A1=activation["A1"]
    A2=activation["A2"]
    W2=para["W2"]
    m= Y.shape[1]
    dZ2 = A2 -Y         # shape : (1, 100)
    dW2 = 1/m*np.dot(dZ2,A1.T)      # shape : (1, n1)
    db2 = 1/m*np.sum(dZ2, axis=1 , keepdims=True)       # shape : (1, 1)
    dZ1 = np.dot(W2.T,dZ2)*A1*(1-A1)        # shape : (n1, 100)
    
    dW1 = 1/m * np.dot(dZ1, X.T)        # shape : (n1, 2)
    db1 = 1/m*np.sum(dZ1, axis=1 , keepdims=True)       # shape : (n1, 1)
    gradient={'dW1':dW1,'db1':db1,'dW2':dW2,'db2':db2}
    return gradient
  
def upwardpropagation(gradient, parameters,learningrate):
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    dW1 = gradient["dW1"]
    db1 = gradient["db1"]
    dW2 = gradient["dW2"]
    db2 = gradient["db2"]
    W1 = W1 - learningrate*dW1
    b1 = b1 - learningrate*db1
    W2 = W2 - learningrate*dW2
    b2 = b2 - learningrate*db2
    para={'W1':W1,"b1": b1,'W2':W2,"b2": b2}
    return parameters
    
def prediction(X,parameters):
    activation = fp(X, parameters)
    A2= activation['A2']
    return A2>=0.5

def neuralnetwork(Xtrain,Ytrain,n1=15,learningrate=0.5,numberofiteration=1000):
    
    n0=Xtrain.shape[0]
    n2=1
    parameters=initialisation(n1, n0, n2)
    
    trainloss=[]
    trainacc=[]
    for i in range(numberofiteration):
        acti= fowardpropagation(Xtrain, parameters)
        grad=backpropagation(Xtrain, Ytrain, activation, parameters)
        para= upwardpropagation(gradient, parameters, learningrate)
        # every 10 iteration
        if i%10 == 0: 
            trainloss.append(logloss(activation["A2"], Ytrain))
            Yprediction = prediction(Xtrain, parameters)
            currentaccuracy = accuracy_score(Ytrain.flatten(),Yprediction.flatten())
            trainacc.append(currentaccuracy)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(trainloss, label='train loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(trainacc, label='accuracy')
    plt.legend()
    plt.show()

    
    
    
