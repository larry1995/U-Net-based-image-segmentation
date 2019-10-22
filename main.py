from model import *
from data import *
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import os
import cv2
import matplotlib as plt
import tensorflow as tf

data_gen_args = dict(
						
				rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.01,
                    horizontal_flip=True,
                    fill_mode='nearest'
                    
                    )






#image_normalized('data/membrane/train/image_ilker','data/membrane/train/new_image')
#image_normalized('data/membrane/train/masks_ilker','data/membrane/train/new_mask')

#histoequalization('data/membrane/train/training',165)
myGene = trainGenerator(5,'data/membrane/train','image_ilker','masks_ilker',data_gen_args,save_to_dir ="data/membrane/train/augment")
vali = validationGenerator(5,'data/membrane/validation','vali','vali_groundtruth')
#kf = KFold(n_splits = 10,random_state = None, shuffle = True)
#kf.get_
 
model = unet()
#earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=1, mode='auto')

#model_checkpoint = ModelCheckpoint(monitor='val_acc',verbose=1, save_best_only=True)
history = model.fit_generator(myGene,steps_per_epoch=35,epochs=30,verbose = 1,validation_data= vali,validation_steps=5,class_weight = [0.9,0.1],max_queue_size = 1)
##you can lower the steps and epochs to save time##
#history = model.fit_generator(myGene,steps_per_epoch=10,epochs=5,verbose = 1,validation_data= vali,validation_steps=5,class_weight = [0.9,0.1],max_queue_size = 1)

testGene = testGenerator("data/membrane/public_test")
results = model.predict_generator(testGene,20,verbose=1)
#model.save('Une.h5')
#acc = history.history['acc']
#val_acc = history.history['val_acc']
#loss = history.history['loss']
#val_loss = history.history['val_loss']
#epochs = range(1,len(acc)+1)
#plt.plot(epochs,acc,'bo',label = 'Training_acc')
#plt.plot(epochs,val_acc,'b',label ='Validation_acc')
#plt.title("Training and validation accuracy")
#plt.legend()
#plt.figure()



saveResult("data/membrane/public_test",results)
