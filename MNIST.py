#!/usr/bin/env python
# coding: utf-8

# In[12]:



import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K


# In[5]:


#global varible 
batch_size = 128
nb_classes = 10
epochs = 10
#image pixel of mnist dataset is 28*28,RGB is 1
img_h,img_w = 28,28
#define number of kernels is 32
nb_filters = 32
#size of pooling area for max pooling
pool_size = (2,2)
#define kernel size is 3*3
kernel_size = (3,3)


# In[6]:


#split between train and test data
(X_train,y_train),(X_test,y_test) = mnist.load_data()


# In[7]:


#Default keras backend is tensorflow,so define this format in this form
X_train = X_train.reshape(X_train.shape[0],img_h,img_w,1)
X_test = X_test.reshape(X_test.shape[0],img_h,img_w,1)
input_shape = (img_h,img_w,1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print('X_train shape:',X_train.shape)
print(X_train.shape[0],'train samples')
print(X_test.shape[0],'test samples')


# In[8]:


#Convert to one-hot Type
Y_train = np_utils.to_categorical(y_train,nb_classes)
Y_test = np_utils.to_categorical(y_test,nb_classes)


# In[7]:


#Construct model
#Convolution -- Activation --pooling
#border_mode is same in order to get the same output size as the input size
"""model.add(Convolution2D(nb_filters,kernel_size[0],kernel_size[1],border_mode='same',input_shape=input_shape))"""
model = Sequential()
#Convolution Layer 1
model.add(Convolution2D(nb_filters,(kernel_size[0],kernel_size[1]),padding='same',input_shape=input_shape))
#Activation layer
model.add(Activation('relu'))
#Convolution Layer2
model.add(Convolution2D(nb_filters,(kernel_size[0],kernel_size[1])))
#Activation Layer
model.add(Activation('relu'))
#Max_Pooling
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))#Neuronal random inactivation
model.add(Flatten())#Pull into 1 dimensional data
# Fully connected layer
model.add(Dense(128))#Layer 1
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))#Layer 2

model.add(Activation('softmax'))


# In[9]:


model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])


# In[10]:


#Train model
model.fit(X_train,Y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_test,Y_test))


# In[11]:


#Evaluate model
score = model.evaluate(X_test,Y_test,verbose=0)
print('Test score:',score[0])
print('Test accuracy:',score[1])


# In[13]:


#Create another model
model2 = Sequential()

model2.add(Convolution2D(nb_filters,(5,5),padding='same',input_shape=input_shape))
model2.add(BatchNormalization(axis=-1))
model2.add(Activation('linear'))
model2.add(Convolution2D(nb_filters,(5,5)))
model2.add(Activation('linear'))
model2.add(MaxPooling2D(pool_size=pool_size))

model2.add(Convolution2D(64,(kernel_size[0],kernel_size[1])))
model2.add(BatchNormalization(axis=-1))
model2.add(Activation('linear'))
model2.add(Convolution2D(64,(kernel_size[0],kernel_size[1])))
model2.add(BatchNormalization(axis=-1))
model2.add(Activation('linear'))
model2.add(MaxPooling2D(pool_size=pool_size))

model2.add(Flatten())

#Fully connected layer
model2.add(Dense(256))
model2.add(BatchNormalization())
model2.add(Activation('linear'))
model2.add(Dropout(0.2))
model2.add(Dense(128))
model2.add(Activation('relu'))
model2.add(Dropout(0.5))
model2.add(Dense(nb_classes))#Layer 3

model2.add(Activation('softmax'))         


# In[15]:


model2.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])


# In[16]:


model2.fit(X_train,Y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_test,Y_test))


# In[18]:


#Evaluate model
score = model2.evaluate(X_test,Y_test,verbose=0)
print('Test score:',score[0])
print('Test accuracy:',score[1])

