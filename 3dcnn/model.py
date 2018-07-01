

import cv2
import os
import numpy as np
import pandas as pd
All_frame=[]
base=r'/home/dgxuser103/Surbhi/9/Surbhi/data/newdatacnn/no'
i=0
k=0
z=0
framez1=[]

for file in os.listdir(base):
   img=cv2.imread(os.path.join(base,file))
   gray=cv2.resize(img,(132,132))
   framez1.append(gray)
   k=k+1 
   if(k==16):
       ipt=np.array(framez1)
       All_frame.append(ipt)
       framez1=[]
       k=0

# z=z+1
x1=len(All_frame)    
   
base=r'/home/dgxuser103/Surbhi/9/Surbhi/data/newdatacnn/down'
i=0
k=0
framez1=[]
#try:
for file in os.listdir(base):
   img=cv2.imread(os.path.join(base,file))
   gray=cv2.resize(img,(132,132))
   framez1.append(gray)
   k=k+1 
   if(k==16):
       ipt=np.array(framez1)
       All_frame.append(ipt)
       framez1=[]
       k=0
#except:
 #z=z+1
x2=len(All_frame)-x1     

base=r'/home/dgxuser103/Surbhi/9/Surbhi/data/newdatacnn/up'
i=0
k=0
framez1=[]
#try:
for file in os.listdir(base):
   img=cv2.imread(os.path.join(base,file))
   gray=cv2.resize(img,(132,132))
   framez1.append(gray)
   k=k+1 
   if(k==16):
       ipt=np.array(framez1)
       All_frame.append(ipt)
       framez1=[]
       k=0
#except:
 #z=z+1
x3=len(All_frame)-x2-x1     

total_length=x2+x1+x3


label=np.ones((total_length,),dtype = int)
label[0:x1-1]= 0
label[x1:x1+x2-1] = 1
label[x1+x2:]=2

X_tr_array=np.array(All_frame)
train_data = [X_tr_array,label]
(train_set, y_train) = (train_data[0],train_data[1])



print('X_Train shape:', train_set.shape)
patch_size = 16
batch_size = 2
nb_classes = 3
nb_epoch = 16
img_rows=132
img_cols=132

from keras.utils import np_utils, generic_utils


Y_train = np_utils.to_categorical(y_train, nb_classes)
print(Y_train.shape)
from keras import regularizers

#from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
import keras
from keras import optimizers
from keras.layers import Reshape
from keras.layers import LSTM
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras import backend as K
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import cross_validation
from keras.layers.normalization import BatchNormalization
from sklearn import preprocessing
from keras.models import model_from_json

from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
train_set = train_set.astype('float32')

train_set -= np.mean(train_set)

train_set /=np.max(train_set)

print(train_set.shape)
#train=np.rollaxis(train_set,1,5)
#print(train.shape)
model = Sequential()
        # 1st layer group
model.add(Convolution3D(64, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv1',
                         subsample=(1, 1, 1),
                         input_shape=(patch_size,img_rows,img_cols,3)))
model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                               border_mode='valid', name='pool1'))
        # 2nd layer group
model.add(Convolution3D(128, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv2',
                         subsample=(1, 1, 1)))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool2'))
        # 3rd layer group
model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv3a',
                         subsample=(1, 1, 1)))
model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv3b',
                         subsample=(1, 1, 1)))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool3'))
        # 4th layer group
model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv4a',
                         subsample=(1, 1, 1)))
model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv4b',
                         subsample=(1, 1, 1)))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool4'))

        # 5th layer group
model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv5a',
                         subsample=(1, 1, 1)))
model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv5b',
                         subsample=(1, 1, 1)))
model.add(ZeroPadding3D(padding=(0, 1, 1)))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool5'))
model.add(Flatten())
model.add(Dropout(0.5))
# FC layers group
model.add(Dense(2048, activation='relu', name='fc6'))
model.add(Dropout(0.5))
model.add(Reshape((16,128),input_shape=(2048,)))
model.add(LSTM(2048,return_sequences=False,dropout=0.2))

#model.add(reshape(

model.add(Dense(3, activation='softmax'))
#

sgd=optimizers.SGD(lr=0.003,momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
for layer in model.layers:
  print(layer.get_output_at(0).get_shape().as_list())
print(model.summary())


print(Y_train.shape)

path=r'/raid/research/data/Dr.Vinit/Interns/9/checkpo4.hdf5'
checkpoint=ModelCheckpoint(path,monitor='val_acc',verbose=1,save_best_only=True,mode='max')
model_json = model.to_json()
with open("/home/dgxuser103/Surbhi/9/Surbhi/data/cnnmodel.json", "w") as json_file:
    json_file.write(model_json)
#model.load_weights(r'/home/dgxuser103/Surbhi/9/Surbhi/data/newweight5.hdf5')

class decay_lr(Callback):
    
    n_epoch = 4
    decay = 0.1
    
    def _init_(self):
        super(decay_lr, self)._init_()
        self.n_epoch=4
        self.decay=0.1

    def on_epoch_begin(self, epoch, logs={}):
        print(K.eval(self.model.optimizer.lr))
        old_lr = K.eval(self.model.optimizer.lr)
        if epoch > 1 and epoch%self.n_epoch == 0 :
            new_lr= self.decay*old_lr
            K.set_value(self.model.optimizer.lr, new_lr)
        else:
            K.set_value(self.model.optimizer.lr, old_lr)
decaySchedule=decay_lr()
X_train_new, X_val_new, y_train_new,y_val_new =  train_test_split(train_set, Y_train, test_size=0.2, random_state=4)
print(X_train_new.shape)
calls=[decaySchedule]
#model.load_weights(r'/home/dgxuser103/Surbhi/9/Surbhi/data/3dcnn16epoch.hdf5')


def visualizeHis(hist):
    # visualizing losses and accuracy

    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['acc']
    val_acc=hist.history['val_acc']
    xc=range(nb_epoch)

    loss=plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train','val'])
    print("plotting")
    acc=plt.figure(2,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)

    #plt.show()
    loss.savefig(r'/home/dgxuser103/Surbhi/9/Surbhi/data/3dcnnadamloss16.png')
    acc.savefig(r'/home/dgxuser103/Surbhi/9/Surbhi/data/3dcnnadamacc16.png')

hist = model.fit(X_train_new, y_train_new, validation_data=(X_val_new,y_val_new),batch_size=batch_size,callbacks=calls,verbose=1,nb_epoch = nb_epoch,shuffle=True)
fname = r'/home/dgxuser103/Surbhi/9/Surbhi/data/3dcnn100epoch.hdf5'
model.save_weights(fname,overwrite=True)
visualizeHis(hist)