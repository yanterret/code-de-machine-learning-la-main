import torch
import torch.nn as nn
import mitdeeplearning as mdl
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.datasets import  make_circles

X, Y =  make_circles(n_samples=100, noise=0.3, factor=0.3, random_state=0)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='summer')
Y = Y.reshape((Y.shape[0], 1))    
     
Numberiteration=1000
lossfonction = nn.BCELoss()

x= torch.tensor(X,dtype=torch.float32)
y=torch.tensor(Y, dtype=torch.float32)
lossvisualization=[]

model = nn.Sequential(
    nn.Linear(2,10),nn.ReLU(),
    nn.Linear(10,10),nn.ReLU(),
    nn.Linear(10,1),nn.Sigmoid(),  
    )

optimizer = optim.Adam(model.parameters(), lr=0.001)
 
for i in range(Numberiteration):
    yprediction = model(x)
    loss = lossfonction(yprediction,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i%10 == 0: 
        lossvisualization.append(float(loss))
        
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(lossvisualization, label='train loss')
plt.legend()
plt.show()
