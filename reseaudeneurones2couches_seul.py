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

def ini(n1, n0, n2):
    
    m= Y.shape[1]
    W1  = rd.randn(n1 , n0)*(1 / m)**(-1/2)
    b1  = rd.randn(n1 , 1)
    W2  = rd.randn(n2 , n1)*(1 / m)**(-1/2)
    b2  = rd.randn(n2 , 1)
    para={'W1':W1,"b1": b1,'W2':W2,"b2": b2}
    return para

def fp(X, para):
    
    W1 = para["W1"]
    b1 = para["b1"]
    W2 = para["W2"]
    b2 = para["b2"]
    Z1 = np.dot(W1,X) + b1      # shape : (n1, 100)
    A1=1/(1+np.exp(-Z1))        # shape : (n1, 100)
    Z2 = np.dot(W2,A1) + b2         # shape : (1, 100)
    A2=1/(1+np.exp(-Z2))           # shape : (1, 100)
    
    acti={"A1":A1,"A2":A2}
    
    return acti 

# cout logloss -1/m*sum(Ylog(A)+(1-Y)*log(1-A)))
#  Z = WX+b ou Z= WA1+b

def bp(X,Y,acti,para):
    
    A1=acti["A1"]
    A2=acti["A2"]
    W2=para["W2"]
    m= Y.shape[1]
    dZ2 = A2 -Y         # shape : (1, 100)
    dW2 = 1/m*np.dot(dZ2,A1.T)      # shape : (1, n1)
    db2 = 1/m*np.sum(dZ2, axis=1 , keepdims=True)       # shape : (1, 1)
    dZ1 = np.dot(W2.T,dZ2)*A1*(1-A1)        # shape : (n1, 100)
    
    dW1 = 1/m * np.dot(dZ1, X.T)        # shape : (n1, 2)
    db1 = 1/m*np.sum(dZ1, axis=1 , keepdims=True)       # shape : (n1, 1)
    grad={'dW1':dW1,'db1':db1,'dW2':dW2,'db2':db2}
    return grad
  
def up(grad, para,learningrate):
    
    W1 = para["W1"]
    b1 = para["b1"]
    W2 = para["W2"]
    b2 = para["b2"]
    dW1 = grad["dW1"]
    db1 = grad["db1"]
    dW2 = grad["dW2"]
    db2 = grad["db2"]
    W1 = W1 - learningrate*dW1
    b1 = b1 - learningrate*db1
    W2 = W2 - learningrate*dW2
    b2 = b2 - learningrate*db2
    para={'W1':W1,"b1": b1,'W2':W2,"b2": b2}
    return para
    
def predi(X,para):
    acti = fp(X, para)
    A2= acti['A2']
    return A2>=0.5

def nn(Xt,Yt,n1=10,learningrate=0.1,niter=1000):
    
    n0=Xt.shape[0]
    n2=1
    para=ini(n1, n0, n2)
    
    trainloss=[]
    trainacc=[]
    for i in range(niter):
        acti= fp(Xt, para)
        grad=bp(Xt, Yt, acti, para)
        para= up(grad, para, learningrate)
        
        if i%10 == 0:
            trainloss.append(logloss(acti["A2"], Yt))
            ypred = predi(Xt, para)
            currentaccuracy = accuracy_score(Yt.flatten(),ypred.flatten())
            trainacc.append(currentaccuracy)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(trainloss, label='train loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(trainacc, label='accuracy du modele')
    plt.legend()
    plt.show()

def plot_decision_boundary(model, X, Y):
    # X est de forme (2, 100) donc on le remet en (100, 2) pour le meshgrid
    X_orig = X.T
    x_min, x_max = X_orig[:, 0].min() - 1, X_orig[:, 0].max() + 1
    y_min, y_max = X_orig[:, 1].min() - 1, X_orig[:, 1].max() + 1
    h = 0.01  # step size in the mesh

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # reshape (grid_points, 2) -> transpose to (2, grid_points)
    grid = np.c_[xx.ravel(), yy.ravel()].T

    Z = model(grid)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap='summer', alpha=0.5)
    plt.scatter(X_orig[:, 0], X_orig[:, 1], c=Y.flatten(), cmap='summer', edgecolors='k')
    plt.title("Frontière de décision")
    plt.show()
    
    return para
    
para = nn(X, Y)
def plot_decision_boundary(model, X, Y):
    # X est de forme (2, 100) donc on le remet en (100, 2) pour le meshgrid
    X_orig = X.T
    x_min, x_max = X_orig[:, 0].min() - 1, X_orig[:, 0].max() + 1
    y_min, y_max = X_orig[:, 1].min() - 1, X_orig[:, 1].max() + 1
    h = 0.01  # step size in the mesh

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # reshape (grid_points, 2) -> transpose to (2, grid_points)
    grid = np.c_[xx.ravel(), yy.ravel()].T

    Z = model(grid)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap='summer', alpha=0.5)
    plt.scatter(X_orig[:, 0], X_orig[:, 1], c=Y.flatten(), cmap='summer', edgecolors='k')
    plt.title("Frontière de décision")
    plt.show()

plot_decision_boundary(lambda x: predi(x, para), X, Y)
         
    
    
    
