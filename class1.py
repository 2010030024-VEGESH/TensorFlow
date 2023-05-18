import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)

# Initialization of Tensors
# Tensors are multidimensional arrays with a uniform data type

x = tf.constant(4)
print(x)

x = tf.constant(4, shape = (1,1), dtype = tf.float32)
print(x)

x = tf.constant(4.0)
print(x)

x = tf.constant([[1,2,3],[4,5,6]]) #it creates a tensor with 2 rows and 3 cols
print(x)

x = tf.ones((3,3))
print(x)

x = tf.zeros((2,3))
print(x)

x = tf.eye(3) #it creates a identity matxix of size 3*3
print(x)


x = tf.random.normal((3,3),mean = 0 ,stddev = 1)
print(x)
x = tf.random.uniform((2,3),minval = 0,maxval = 1)

x = tf.range(9)
print(x)

xx = tf.range(start=1,limit = 10,delta=2)
print(xx)

x = tf.cast(xx,dtype=tf.float64)
print(x)

x = tf.ones((4,3))
print(x)
y = tf.cast(x,dtype = tf.float64)
print(y)
#Mathematical Operations

x = tf.constant([1,2,3])
y = tf.constant([7,5,6])
z = tf.add(x,y)
print(z)
zz = x+y
print(zz)
veg = tf.subtract(x, y)
print(veg)
z = x-y
print(z)

z = tf.divide(x,y)
print(z)
z = x/y
print(z)
z = tf.multiply(x,y)
print(z)
x = x*y
print(z)

z = tf.tensordot(x,y, axes = 1)
print(z)
print('lol')
x = tf.random.normal((2,3))
print(x)
y = tf.random.normal((3,4))
print(y)
z = tf.matmul(x,y)
print(z)
z = x @ y
print(z)

x = tf.constant([0,1,1,2,3,1,2,3])
print(x[:])
print(x[1:])
print(x[1:4])
print(x[::-1])
print(x[::2])

# indices = tf.constant([0,3])
# x_ind = tf.gather(x, indices)
# print(x_ind)
x = tf.constant([[1,2],
                 [3,4],
                 [5,6]])
print(x[0,:])
print(x[0:2,:])

#Indexing

#Reshaping
print('reshaping')
x = tf.range(9)
print(x)
x = tf.reshape(x,(3,3))
print(x)

x = tf.transpose(x,perm=[1,0])
print(x)