import tensorflow as tf
import numpy as np

#创建数据
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.25

Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))#设置权重变量并随机为-1.0~1.0 之间的数
biases = tf.Variable(tf.zeros([1]))#设置偏置变量为0

y = Weights*x_data + biases#预测数据

loss = tf.reduce_mean(tf.square(y-y_data))#计算损失（偏差）使用方差的形式

optimizer = tf.train.GradientDescentOptimizer(0.5) #0.5 是学习率
train = optimizer.minimize(loss)#训练函数（使loss减小）

init = tf.global_variables_initializer()#初始化变量

sess = tf.Session()#设置会话
sess.run(init)#通过会话开启初始化程序

for step in range(201):#训练200次
    if step % 20 == 0:
        print(step,sess.run(Weights),sess.run(biases),sess.run(loss))#每20步打印步数，权重，损失（注意：查看tf内的变量必须用会话）
    sess.run(train)#通过会话进行训练