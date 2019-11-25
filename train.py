import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import numpy as np
import cv2
from sklearn.utils import shuffle

#Load Images from Swing
loadedImages = []
for i in range(0, 380):
    image = cv2.imread('Dataset/A/a_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(97, 100, 1))

#Load Images From Palm
for i in range(0, 380):
    image = cv2.imread('Dataset/B/b_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(97, 100, 1))
    
#Load Images From Fist
for i in range(0, 380):
    image = cv2.imread('Dataset/C/c_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(97, 100, 1))
    
for i in range(0, 380):
    image = cv2.imread('Dataset/D/d_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(97, 100, 1))

for i in range(0, 380):
    image = cv2.imread('Dataset/E/e_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(97, 100, 1))
# import pdb; pdb.set_trace()

# Create OutputVector

outputVectors = []
for i in range(0, 380):
    outputVectors.append([1, 0, 0, 0,0])

for i in range(0, 380):
    outputVectors.append([0, 1, 0,0,0])

for i in range(0, 380):
    outputVectors.append([0, 0, 1,0,0])
for i in range(0, 380):
    outputVectors.append([0, 0, 0,1,0])
for i in range(0, 380):
    outputVectors.append([0, 0, 0,0,1])




testImages = []

#Load Images for swing


#Load Images for Palm
for i in range(380, 400):
    image = cv2.imread('Dataset/A/a_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(97, 100, 1))
for i in range(380, 400):
    image = cv2.imread('Dataset/B/b_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(97, 100, 1))
for i in range(380, 400):
    image = cv2.imread('Dataset/C/c_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(97, 100, 1))
for i in range(380, 400):
    image = cv2.imread('Dataset/D/d_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(97, 100, 1))
for i in range(380, 400):
    image = cv2.imread('Dataset/E/e_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(97, 100, 1))

testLabels = []

for i in range(0, 20):
    testLabels.append([1, 0, 0,0,0])
    
for i in range(0, 20):
    testLabels.append([0, 1, 0,0,0])

for i in range(0, 20):
    testLabels.append([0, 0, 1,0,0])
for i in range(0, 20):
    testLabels.append([0, 0, 0,1,0])
for i in range(0, 20):
    testLabels.append([0, 0, 0,0,1])


# Define the CNN Model
# Define the CNN Model
tf.reset_default_graph()
convnet=input_data(shape=[None,97,100,1],name='input')
convnet=conv_2d(convnet,64,3,activation='relu')
convnet=conv_2d(convnet,64,3,activation='relu')
convnet=max_pool_2d(convnet,3,strides=2)
convnet=conv_2d(convnet,128,3,activation='relu')
convnet=conv_2d(convnet,128,3,activation='relu')
convnet=max_pool_2d(convnet,3,strides=2)

convnet=conv_2d(convnet,256,3,activation='relu')
convnet=conv_2d(convnet,256,3,activation='relu')
convnet=conv_2d(convnet,256,3,activation='relu')
convnet=max_pool_2d(convnet,3,strides=2)

convnet=conv_2d(convnet,512,3,activation='relu')
convnet=conv_2d(convnet,512,3,activation='relu')
convnet=conv_2d(convnet,512,3,activation='relu')
convnet=max_pool_2d(convnet,3,strides=2)

convnet=conv_2d(convnet,256,3,activation='relu')
convnet=conv_2d(convnet,256,3,activation='relu')
convnet=conv_2d(convnet,256,3,activation='relu')
convnet=max_pool_2d(convnet,3,strides=2)

convnet=fully_connected(convnet,128,activation='relu')
convnet=fully_connected(convnet,64,activation='relu')
convnet=dropout(convnet,0.75)

convnet=fully_connected(convnet,5,activation='softmax')

convnet=regression(convnet,optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy',name='regression')

model=tflearn.DNN(convnet,tensorboard_verbose=0)

class EarlyStoppingCallback(tflearn.callbacks.Callback):
    def __init__(self, val_acc_thresh,val_loss):
        """ Note: We are free to define our init function however we please. """
        self.val_acc_thresh = val_acc_thresh
        self.loss=val_loss

    def on_epoch_end(self, training_state):
        """ """
        # Apparently this can happen.
        if training_state.val_acc is None or training_state.val_loss is None: return
        if training_state.val_acc > self.val_acc_thresh and training_state.val_loss<self.loss:
            raise StopIteration

# Initializae our callback.
early_stopping_cb = EarlyStoppingCallback(val_acc_thresh=0.99,val_loss=0.01)

print(len(loadedImages),len(outputVectors))
# Shuffle Training Data
loadedImages, outputVectors = shuffle(loadedImages, outputVectors, random_state=0)

# Train model

history=model.fit(loadedImages, outputVectors, n_epoch=50,callbacks=early_stopping_cb,
           validation_set = (testImages, testLabels),
           snapshot_step=100, show_metric=True, run_id='convnet_coursera')
print(history)

model.save("TrainedModel//ABCDE.h5")





# Shuffle Training Data
loadedImages, outputVectors = shuffle(loadedImages, outputVectors, random_state=0)

# Train model
model.fit(loadedImages, outputVectors, n_epoch=50,
           validation_set = (testImages, testLabels),
           snapshot_step=100, show_metric=True, run_id='convnet_coursera')

model.save("TrainedModel/test.h5")