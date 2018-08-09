import tensorflow as tf
import numpy as np

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],
                       [2]])
product = tf.matmul(matrix1,matrix2)

#method1
sess = tf.Session()
result = sess.run(product)
print(result,'1')
sess.close()
#[[12]]

#method2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2,'2')