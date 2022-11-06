from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose
from tensorflow.keras import Sequential


def model_res(input_size):
    model = Sequential()
    model.add(Conv2D(16,(3,3),padding='same',activation='relu',input_shape=input_size))
    model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
    model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
    model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
    model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(100,(3,3),padding='same',activation='relu'))
    model.add(Conv2D(100,(3,3),padding='same',activation='relu'))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
    model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
    model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
    model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
    model.add(Conv2D(16,(3,3),padding='same',activation='relu'))
    model.add(Conv2D(3,(3,3),padding='same',activation='relu'))
    
    return model
    



                     
    