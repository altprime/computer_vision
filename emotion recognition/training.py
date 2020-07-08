# using vgg-16 architecture

from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPooling2D
import os

num_classes = 7 # angry, sad, happy, neutral, surprised
imgRows, imgCols = 48, 48
batch_size = 32 # no of images at a

"""

                    *****     Uncomment this section     *****
                    
train_dir = <insert path for traing dataset>
validation_dir = "<insert path for validation dataset>

"""

train_datagenerator = ImageDataGenerator(
    rescale= 1./255, #noramlise each pixel in image
    rotation_range= 30, # rotate 30 deg left and right to gen multiple images from one image
    shear_range= 0.3, zoom_range= 0.3,
    width_shift_range= 0.4, height_shift_range= 0.4, horizontal_flip= True,
    fill_mode= 'nearest' # fill moved pixels from the neaerest pixels
)

validation_datagenerator = ImageDataGenerator(rescale= 1./255)

train_gen = train_datagenerator.flow_from_directory(
    train_dir,
    color_mode= 'grayscale', # colour not required for emotion detection
    target_size= (imgRows, imgCols), batch_size= batch_size,
    class_mode= 'categorical', shuffle= True
)

validation_gen = validation_datagenerator.flow_from_directory(
    validation_dir,
    color_mode= 'grayscale', # colour not required for emotion detection
    target_size= (imgRows, imgCols), batch_size= batch_size,
    class_mode= 'categorical', shuffle= True
)

# neural network
# first block
model= Sequential()
model.add(Conv2D(32, (3,3), padding= 'same', kernel_initializer= 'he_normal', input_shape=(imgRows, imgCols,1)))
model.add((Activation('elu')))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), padding= 'same', kernel_initializer= 'he_normal', input_shape=(imgRows, imgCols,1)))
model.add((Activation('elu')))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size= (2,2)))
model.add(Dropout(0.2))
# second block
model.add(Conv2D(64, (3,3), padding= 'same', kernel_initializer= 'he_normal'))
model.add((Activation('elu')))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding= 'same', kernel_initializer= 'he_normal'))
model.add((Activation('elu')))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size= (2,2)))
model.add(Dropout(0.2))
# third block
model.add(Conv2D(128, (3,3), padding= 'same', kernel_initializer= 'he_normal'))
model.add((Activation('elu')))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), padding= 'same', kernel_initializer= 'he_normal'))
model.add((Activation('elu')))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size= (2,2)))
model.add(Dropout(0.2))
# fourth block
model.add(Conv2D(256, (3,3), padding= 'same', kernel_initializer= 'he_normal'))
model.add((Activation('elu')))
model.add(BatchNormalization())
model.add(Conv2D(256, (3,3), padding= 'same', kernel_initializer= 'he_normal'))
model.add((Activation('elu')))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size= (2,2)))
model.add(Dropout(0.2))
# fifth block
model.add(Flatten())
model.add(Dense(64, kernel_initializer= 'he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
# sixth block
model.add(Dense(64, kernel_initializer= 'he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
# final
model.add(Dense(num_classes, kernel_initializer= 'he_normal'))
model.add(Activation('softmax'))

print(model.summary())



from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
checkpoint = ModelCheckpoint('Emotion_vgg.h5',
                             monitor= 'val_loss',
                             mode= 'min',
                             save_best_only= True,
                             verbose= 1)
early_stopping = EarlyStopping(monitor= 'val_loss',
                               min_delta= 0,
                               patience= 3,
                               verbose= 1,
                               restore_best_weights= True)
reduce_learingRate = ReduceLROnPlateau(monitor= 'val_loss',
                                       factor= 0.2,
                                       patience= 3,
                                       verbose= 1,
                                       min_delta= 0.0001)
callback = [early_stopping, checkpoint, reduce_learingRate]

model.compile(loss = 'categorical_crossentropy',
              optimizer = Adam(lr = 0.001),
              metrics = ['accuracy'])
nb_train_samples = 28709
nb_validation_samples = 3589
epochs = 25

history = model.fit_generator(
    train_gen,
    steps_per_epoch= nb_train_samples//batch_size,
    epochs = epochs,
    callbacks = callback,
    validation_data= validation_gen,
    validation_steps= nb_validation_samples//batch_size)


