import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

x_data = np.linspace(-4,4,300,dtype=np.float32)[:,np.newaxis]
y_data = np.linspace(-4,4,300,dtype=np.float32)[:,np.newaxis]
noise = np.random.normal(0,0.5,x_data.shape).astype(np.float32)
z_data = np.sqrt(x_data**2  + y_data**2 )
inputs = np.hstack((x_data,y_data))

ins = tf.placeholder(tf.float32,[None,2])
zs = tf.placeholder(tf.float32,[None,1])

l1 = add_layer(ins,2,4,tf.nn.tanh)
prediction=add_layer(l1,4,1,None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(zs-prediction),reduction_indices=[1]))
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

fig = plt.figure()
ax = Axes3D(fig)
X,Y = np.meshgrid(np.linspace(-4,4,300,dtype=np.float32),np.linspace(-4,4,300,dtype=np.float32))
print(X,Y)
ax.plot_surface(X,Y,z_data, rstride=10, cstride=10, cmap=plt.get_cmap('rainbow'))
plt.show()