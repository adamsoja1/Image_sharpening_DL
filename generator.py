import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
#generator
def image_load_generator_x(path,batch_size):
    files = os.listdir(f'{path}')
    L = len(files)
    while True:
        batch_start = 0
        batch_size_end = batch_size
        while batch_start < L:
            limit = min(batch_size_end,L)
            
            files_batched = files[batch_start:limit]
            
            #loading data
            x_train = []
            
            for file in files_batched:
                X_train = np.load(f'{path}/{file}')
            
                x_train.append(X_train)

            

            x_train = np.array(x_train)
            x_train = x_train/255
            
            yield(x_train)
            
            batch_start +=batch_size
            batch_size_end +=batch_size
            
            
 # generator blurry           
def image_load_generator_blurry(path,batch_size):

    files = os.listdir(f'{path}')
    L = len(files)
    while True:
        batch_start = 0
        batch_size_end = batch_size
        while batch_start < L:
            limit = min(batch_size_end,L)
            
            files_batched = files[batch_start:limit]
            
            #loading data
            x_train = []
            
            for file in files_batched:
                X_train = np.load(f'{path}/{file}')
                

                image = cv2.GaussianBlur(X_train, (5, 5), cv2.BORDER_DEFAULT)
                
  
                x_train.append(image)

            x_train = np.array(x_train)
            x_train = x_train/255
            
            yield(x_train)
            
            
            batch_start +=batch_size
            batch_size_end +=batch_size
            