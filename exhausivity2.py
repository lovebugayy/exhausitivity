import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]

def forward(x):
    return w*x+b

def Loss(x,y):
    y_pred=forward(x)
    return (y-y_pred)*(y-y_pred)

w_list=[]
b_list=[]
loss_list=[]
for w in np.arange(0.0,4.1,0.1):
    for b in np.arange(-2.0,2.1,0.1):
        print('w=',w)
        print('b=',b)
        l_sum=0
        w_list.append(w)
        b_list.append(b)
        for x_val,y_val in zip(x_data,y_data):
            y_pred_val=forward(x_val)
            loss_val=Loss(x_val,y_val)
            l_sum+=loss_val
            print('\t',x_val,y_val,y_pred_val,loss_val)
        loss_list.append(l_sum)
        print('MSE=',l_sum/3)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(w_list,b_list,loss_list)
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('Loss')
plt.show()
