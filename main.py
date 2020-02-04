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

#histoequalization('data/membrane/raw',20)
myGene = trainGenerator(3,'data/membrane/train','raw','mitochondria',data_gen_args,save_to_dir ="data/membrane/train/augment")
vali = validationGenerator(5,'data/membrane/validation','testing','testing_groundtruth')


 
model = unet()
earlyStopping=EarlyStopping(monitor='val_loss', patience=80, verbose=1, mode='auto')
filepath="weights.best.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,mode='max')
callbacks_list = [checkpoint]
#earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=1, mode='auto')

#model_checkpoint = ModelCheckpoint(monitor='val_acc',verbose=1, save_best_only=True)
#history = model.fit_generator(myGene,steps_per_epoch=35,epochs=100,verbose = 1,validation_data= vali,validation_steps=5,class_weight = [0.9,0.1],max_queue_size = 1)
history = model.fit_generator(myGene,steps_per_epoch=9,epochs=130,verbose = 1,validation_data= vali,validation_steps=5,max_queue_size = 1,callbacks=[checkpoint])
#history = model.fit_generator(myGene,steps_per_epoch=10,epochs=130,verbose = 1,validation_data= vali,class_weight=[0.9,0.1],validation_steps=5,max_queue_size = 1)

testGene = testGenerator("data/membrane/055_HIPPO")
#results = model.predict_generator(testGene,11,verbose=1)
results = model.predict_generator(testGene,23,verbose=1)



saveResult("data/membrane/055_HIPPO",results)
post_processing("data/membrane/055_HIPPO","data/membrane/testdata_processed",21) 
