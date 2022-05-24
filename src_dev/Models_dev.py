#file which contains neural network models and loss functions

#v4_1: includes a second network with ReLU function as last layer

import os
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import tensorflow.keras.backend as K
import math
import h5py


def generator_kj(latent_size, keras_dformat='channels_last', leaky_last_step=False):
    #keras_dformat='channels_first'
    if keras_dformat == 'channels_first':
        axis = 1  #if channels first, -1 if channels last
        dshape = [1, 51, 51, 25]
    else:
        axis = -1
        dshape = [51, 51, 25, 1]
    dim = (11,11,11)

    latent = tf.keras.Input(shape=(latent_size ), dtype="float32")    
    x = tf.keras.layers.Dense(dim[0]*dim[1]*dim[2], input_dim=latent_size)(latent)
    x = tf.keras.layers.Reshape(dim) (x) 
    
    x1 = x
    x2 = tf.keras.layers.Permute([3,1,2])(x)   #permute starts indexing with 1
    x3 = tf.keras.layers.Permute([2,3,1])(x)

    def path_231(x):    # OUT: (51,25,51)
        #1.Conv Block
        x = tf.keras.layers.Conv2D(32, (5, 5), data_format=keras_dformat, use_bias=False, padding='same')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
        x = tf.keras.layers.Dropout(0.2)(x)
        # output shape: (None, 11, 11, 32) - if channels last
        #2.Conv Block
        x = tf.keras.layers.Conv2DTranspose(64, (5,5), strides =(3,2), data_format=keras_dformat, padding="same") (x)
        x = tf.keras.layers.Conv2D(64, (5, 5), data_format=keras_dformat, padding='same', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
        x = tf.keras.layers.Dropout(0.2)(x)
        # output shape: (None, 33, 22, 64)
        #3.Conv Block
        x = tf.keras.layers.Conv2DTranspose(64, (5,5), strides =(2,2), data_format=keras_dformat, padding="same") (x)
        x = tf.keras.layers.Conv2D(64, (7, 8), data_format=keras_dformat, padding='valid', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
        x = tf.keras.layers.Dropout(0.2)(x)
        # output shape: (None, 60, 37, 64)
        # 4. Added block
        x = tf.keras.layers.Conv2D(64, (5, 6), data_format=keras_dformat, padding='valid', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
        x = tf.keras.layers.Dropout(0.2)(x)
        # output shape: (None, 56, 32, 64)
        # 5. Conv Block
        x = tf.keras.layers.Conv2D(64, (4, 5), data_format=keras_dformat, padding='valid', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
        x = tf.keras.layers.Dropout(0.2)(x)
        # output shape: (None, 53, 28, 64)
        # 6. Conv Block
        x = tf.keras.layers.Conv2D(64, (3, 4), data_format=keras_dformat, padding='same', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
        x = tf.keras.layers.Dropout(0.2) (x)
        # output shape: (None, 53, 28, 64)
        # 7. Conv Block
        x = tf.keras.layers.Conv2D(51, (2, 3), data_format=keras_dformat, padding='valid', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x)  
        x = tf.keras.layers.Dropout(0.2) (x)
        # output shape: (None, 52, 26, 64)
        # 8. Conv Block
        x = tf.keras.layers.Conv2D(51, (2, 2), data_format=keras_dformat, padding='valid', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x)  
        x = tf.keras.layers.Dropout(0.2) (x)
        # output shape: (None, 51, 25, 51)
        return x
    

    def path_312(x):    # OUT: (25,51,51)
        #path1
        #1.Conv Block
        x = tf.keras.layers.Conv2D(32, (5, 5), data_format=keras_dformat, use_bias=False, padding='same')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
        x = tf.keras.layers.Dropout(0.2)(x)
        # output shape: (None, 11, 11, 32) - if channels last
        #2.Conv Block
        x = tf.keras.layers.Conv2DTranspose(64, (5,5), strides =(2,3), data_format=keras_dformat, padding="same") (x)
        x = tf.keras.layers.Conv2D(64, (5, 5), data_format=keras_dformat, padding='same', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
        x = tf.keras.layers.Dropout(0.2)(x)
        # output shape: (None, 22, 33, 64)
        #3.Conv Block
        x = tf.keras.layers.Conv2DTranspose(64, (5,5), strides =(2,2), data_format=keras_dformat, padding="same") (x)
        x = tf.keras.layers.Conv2D(64, (8, 7), data_format=keras_dformat, padding='valid', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
        x = tf.keras.layers.Dropout(0.2)(x)
        # output shape: (None, 37, 60, 64)
        # 4. Added block
        x = tf.keras.layers.Conv2D(64, (6, 5), data_format=keras_dformat, padding='valid', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
        x = tf.keras.layers.Dropout(0.2)(x)
        # output shape: (None, 32, 56, 64)
        # 5. Conv Block
        x = tf.keras.layers.Conv2D(64, (5, 4), data_format=keras_dformat, padding='valid', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
        x = tf.keras.layers.Dropout(0.2)(x)
        # output shape: (None, 28, 53, 64)
        # 6. Conv Block
        x = tf.keras.layers.Conv2D(64, (4, 3), data_format=keras_dformat, padding='same', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
        x = tf.keras.layers.Dropout(0.2) (x)
        # output shape: (None, 28, 53, 64)
        # 7. Conv Block
        x = tf.keras.layers.Conv2D(51, (3, 2), data_format=keras_dformat, padding='valid', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x)  
        x = tf.keras.layers.Dropout(0.2) (x)
        # output shape: (None, 26, 52, 64)
        # 8. Conv Block
        x = tf.keras.layers.Conv2D(51, (2, 2), data_format=keras_dformat, padding='valid', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x)  
        x = tf.keras.layers.Dropout(0.2) (x)
        # output shape: (None, 25, 51, 51)
        return x
    
    def path_123(x):    # OUT: (51,51,25)
        #path1
        #1.Conv Block
        x = tf.keras.layers.Conv2D(32, (5, 5), data_format=keras_dformat, use_bias=False, padding='same')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
        x = tf.keras.layers.Dropout(0.2)(x)
        # output shape: (None, 11, 11, 32) - if channels last
        #2.Conv Block
        x = tf.keras.layers.Conv2DTranspose(64, (5,5), strides =(3,3), data_format=keras_dformat, padding="same") (x)
        x = tf.keras.layers.Conv2D(64, (5, 5), data_format=keras_dformat, padding='same', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
        x = tf.keras.layers.Dropout(0.2)(x)
        # output shape: (None, 33, 33, 64)
        #3.Conv Block
        x = tf.keras.layers.Conv2DTranspose(64, (5,5), strides =(2,2), data_format=keras_dformat, padding="same") (x)
        x = tf.keras.layers.Conv2D(64, (7, 7), data_format=keras_dformat, padding='valid', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
        x = tf.keras.layers.Dropout(0.2)(x)
        # output shape: (None, 60, 60, 64)
        # 4. Added block
        x = tf.keras.layers.Conv2D(64, (5, 5), data_format=keras_dformat, padding='valid', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
        x = tf.keras.layers.Dropout(0.2)(x)
        # output shape: (None, 56, 56, 64)
        # 5. Conv Block
        x = tf.keras.layers.Conv2D(64, (4, 4), data_format=keras_dformat, padding='valid', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
        x = tf.keras.layers.Dropout(0.2)(x)
        # output shape: (None, 53, 53, 64)
        # 6. Conv Block
        x = tf.keras.layers.Conv2D(32, (3, 3), data_format=keras_dformat, padding='same', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
        x = tf.keras.layers.Dropout(0.2) (x)
        # output shape: (None, 53, 53, 64)
        # 7. Conv Block
        x = tf.keras.layers.Conv2D(32, (2, 2), data_format=keras_dformat, padding='valid', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x)  
        x = tf.keras.layers.Dropout(0.2) (x)
        # output shape: (None, 52, 52, 32)
        # 8. Conv Block
        x = tf.keras.layers.Conv2D(25, (2, 2), data_format=keras_dformat, padding='valid', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x)  
        x = tf.keras.layers.Dropout(0.2) (x)
        # output shape: (None, 51, 51, 25)
        return x

    x1 = path_123(x1)
    x2 = path_312(x2)
    x3 = path_231(x3)

    # Permute back and stack
    x2 = tf.keras.layers.Permute([2,3,1])(x2)
    x3 = tf.keras.layers.Permute([3,1,2])(x3)
    x = tf.keras.layers.concatenate([x1,x2,x3],axis=axis)
        
#    print(x.shape)
    x = tf.keras.layers.Conv2D(25, (3,3), data_format=keras_dformat, padding='same', use_bias=False,  kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.Reshape(dshape)(x)

    # There has to be ReLU because of the power function (not LeakyReLU)
    x = tf.keras.layers.ReLU() (x)

    return tf.keras.Model(inputs=[latent], outputs=x)


def discriminator_kj(keras_dformat='channels_last', power=0.85):
    if keras_dformat =='channels_last':
        dshape=(51, 51, 25,1)
        dshape_squeezed = dshape[:-1]
        daxis = (1,2,3) # axis for sum
        axis = -1 # channel axis
    elif keras_dformat == 'channels_first':
        dshape=(1, 51, 51, 25)
        dshape_squeezed = dshape[1:]
        daxis=(2,3,4)
        axis = 1
    else:
        pass

    print('disc dshape {}'.format(dshape))
    print('disc dshape_squeezed {}'.format(dshape_squeezed))

    image = tf.keras.layers.Input(shape=dshape, dtype="float32")     #Input Image
    x = image
    x = tf.keras.layers.Reshape(dshape_squeezed)(x) # May need to be changed to dshape[1:]
    
    # LEFT OUT - SHRINKING TO CUBE DESTROYS THE SPATIAL CORRELATIONS BETWEEEN PIXELS AT THE VERY BEGINNING
    # shape = (51,51,25)
    # Shrink to cube 25x25x25
    # x = tf.keras.layers.Conv2D(25, (8, 8), data_format=keras_dformat, use_bias=False, padding='valid')(x)
    # x = tf.keras.layers.Conv2D(26, (8, 8), strides=(2,2), data_format=keras_dformat, use_bias=False, padding='same')(x)
    # shape = (26,26,26)
    # x = tf.keras.layers.Conv2D(26, (8, 8), strides=(2,2), data_format=keras_dformat, use_bias=False, padding='same')(x)
    # x = tf.keras.layers.LeakyReLU() (x)
    # x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
    # x = tf.keras.layers.Dropout(0.2)(x)
    # shape = (26,26,26)

    x1 = x
    # Permute indices of tensors
    x2 = tf.keras.layers.Permute([3,1,2])(x)   #permute starts indexing with 1
    x3 = tf.keras.layers.Permute([2,3,1])(x)   #permute starts indexing with 1
    print("Disc shapes permuted:", x1.shape, x2.shape, x3.shape)

    def path(x):
        #path1
        #1.Conv Block
        x = tf.keras.layers.Conv2D(64, (8, 8), data_format=keras_dformat, use_bias=False, padding='same')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
        x = tf.keras.layers.Dropout(0.2)(x)
        #2.Conv Block
        x = tf.keras.layers.Conv2D(32, (6, 6), data_format=keras_dformat, padding='valid', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x)
        x = tf.keras.layers.Dropout(0.2)(x)
        #x = tf.keras.layers.MaxPooling2D((2, 2),strides=(2,2), data_format=keras_dformat)(x)
        #3.Conv Block
        x = tf.keras.layers.Conv2D(32, (5, 5), data_format=keras_dformat, padding='valid', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x)
        x = tf.keras.layers.Dropout(0.2)(x)
        # shape = (17,17,32)
        #4. Conv Block
        x = tf.keras.layers.Conv2D(32, (4, 4), data_format=keras_dformat, padding='valid', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x)
        x = tf.keras.layers.Dropout(0.2)(x)
        #x = tf.keras.layers.MaxPooling2D((2, 2),strides=(2,2), data_format=keras_dformat)(x)
        #6. Conv Block
        x = tf.keras.layers.Conv2D(32, (3, 3), data_format=keras_dformat, padding='valid', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x)
        x = tf.keras.layers.Dropout(0.2)(x)
        #7. Conv Block
        x = tf.keras.layers.Conv2D(10, (3, 3), data_format=keras_dformat, padding='valid', use_bias=False,  kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.LeakyReLU() (x)
        x = tf.keras.layers.BatchNormalization(axis=axis) (x)
        x = tf.keras.layers.Dropout(0.2)(x)
        #print(x.shape)
        return x

    # Permutations have different shapes - still can go through the same path because the outputs from paths are reshaped into vectors and then concatenated
    print("D path in:", x1.shape, x2.shape, x3.shape)
    x1 = path(x1)
    x2 = path(x2)
    x3 = path(x3)
    print("D path out:", x1.shape, x2.shape, x3.shape)
    
    # Permute indices back
    x2 = tf.keras.layers.Permute([2,3,1])(x2)   #permute starts indexing with 1
    x3 = tf.keras.layers.Permute([3,1,2])(x3)   #permute starts indexing with 1
    
    # x = tf.keras.layers.concatenate([x1,x2,x3],axis=axis) #i stack them on the channels axis

    # Flatten and concatenate the vectors
    x1 = tf.keras.layers.Flatten()(x1)
    x2 = tf.keras.layers.Flatten()(x2)
    x3 = tf.keras.layers.Flatten()(x3)
    x = tf.keras.layers.concatenate([x1,x2,x3],axis=axis)
    
    x = tf.keras.layers.Dense(10, activation='linear')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization(axis=axis)(x)

    #Takes Network outputs as input
    fake = tf.keras.layers.Dense(1, activation='sigmoid', name='generation')(x)   # T/F probab.
    aux = tf.keras.layers.Dense(1, activation='linear', name='auxiliary')(x)       # lin reg on Ep
    #Takes image as input
    inv_image = tf.keras.layers.Lambda(K.pow, arguments={'a':1./power})(image) #get back original image (reverse the power transformation)
    ang = tf.keras.layers.Lambda(ecal_angle, arguments={'daxis':axis})(inv_image) # angle calculation
    ecal = tf.keras.layers.Lambda(ecal_sum, arguments={'daxis':daxis})(inv_image) # sum of energies
    # ecal = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, axis=daxis))(image)    # Florian's implementation of ecal calculation
    
    return tf.keras.Model(inputs=image, outputs=[fake, aux, ang, ecal])
    # return tf.keras.Model(inputs=image, outputs=[fake, aux, ang, ecal, image, inv_image])


def ecal_sum(image, daxis):
    sum = tf.math.reduce_sum(image, axis=daxis) # sum = K.sum(image, axis=daxis)
    return sum

# Calculating angles from images -- called in conditional lambda layer for the discriminator
def ecal_angle(images, daxis, power=1.0):
    # images = prep_image(images, power=1.0)
    images = K.squeeze(images, axis=daxis)

    # size of ecal
    x_shape = images.shape[1]
    y_shape = images.shape[2]
    z_shape = images.shape[3]
    sumtot = tf.math.reduce_sum(images, (1,2,3))# sum of events

    # get 1. where event sum is 0 and 0 elsewhere
    amask = tf.where(tf.math.equal(sumtot, 0.0), tf.ones_like(sumtot) , tf.zeros_like(sumtot))
    masked_events = tf.math.reduce_sum(amask) # counting zero sum events

    # ref denotes barycenter as that is our reference point
    x_ref = tf.math.reduce_sum(tf.math.reduce_sum(images, (2,3)) * (tf.cast(tf.expand_dims(tf.range(x_shape), 0), dtype='float32') + 0.5), axis=1)# sum for x position * x index
    y_ref = tf.math.reduce_sum(tf.math.reduce_sum(images, (1,3)) * (tf.cast(tf.expand_dims(tf.range(y_shape), 0), dtype='float32') + 0.5), axis=1)
    z_ref = tf.math.reduce_sum(tf.math.reduce_sum(images, (1,2)) * (tf.cast(tf.expand_dims(tf.range(z_shape), 0), dtype='float32') + 0.5), axis=1)
    # return max position if sumtot=0 and divide by sumtot otherwise
    x_ref = tf.where(tf.math.equal(sumtot, 0.0), tf.ones_like(x_ref), x_ref/sumtot)
    y_ref = tf.where(tf.math.equal(sumtot, 0.0), tf.ones_like(y_ref), y_ref/sumtot)
    z_ref = tf.where(tf.math.equal(sumtot, 0.0), tf.ones_like(z_ref), z_ref/sumtot)

    # reshape - put in value at the beginning
    x_ref = tf.expand_dims(x_ref, 1)
    y_ref = tf.expand_dims(y_ref, 1)
    z_ref = tf.expand_dims(z_ref, 1)

    sumz = tf.math.reduce_sum(images, axis =(1,2)) # sum for x,y planes going along z

    # Get 0 where sum along z is 0 and 1 elsewhere
    zmask = tf.where(tf.math.equal(sumz, 0.0), tf.zeros_like(sumz) , tf.ones_like(sumz))

    x = tf.expand_dims(tf.range(x_shape), 0) # x indexes
    x = tf.cast(tf.expand_dims(x, 2), dtype='float32') + 0.5
    y = tf.expand_dims(tf.range(y_shape), 0)# y indexes
    y = tf.cast(tf.expand_dims(y, 2), dtype='float32') + 0.5

    # barycenter for each z position
    x_mid = tf.math.reduce_sum(tf.math.reduce_sum(images, axis=2) * x, axis=1)
    y_mid = tf.math.reduce_sum(tf.math.reduce_sum(images, axis=1) * y, axis=1)
    temp = tf.where(tf.math.equal(sumz, 0.0), tf.zeros_like(sumz), tf.ones_like(sumz))  # 0 if sum == 0, 1 if sum != 0
    # print('\ntemp variable')
    # print(temp)
    # print('nonzero elements: {}'.format(tf.math.reduce_sum(temp)))
    # print('\nsumz variable')
    # print(sumz)
    # print('\nx_mid variable')
    # print(x_mid)
    x_mid = tf.where(tf.math.equal(sumz, 0.0), tf.zeros_like(sumz), x_mid/sumz) # if sum != 0 then divide by sum
    # print('\nx_mid variable')
    # print(x_mid)
    y_mid = tf.where(tf.math.equal(sumz, 0.0), tf.zeros_like(sumz), y_mid/sumz) # if sum != 0 then divide by sum

    # Angle Calculations
    z = (tf.cast(tf.range(z_shape), dtype='float32') + 0.5)  * tf.ones_like(z_ref) # Make an array of z indexes for all events
    epsilon = 0.0000007  # replaces k.epsilon(), used as fluff value to prevent /0 errors
    zproj = tf.math.sqrt(tf.math.maximum((x_mid-x_ref)**2.0 + (z - z_ref)**2.0, epsilon))# projection from z axis with stability check
    m = tf.where(tf.math.equal(zproj, 0.0), tf.zeros_like(zproj), (y_mid-y_ref)/zproj)# to avoid divide by zero for zproj =0
    m = tf.where(tf.math.less(z, z_ref),  -1 * m, m)   # sign inversion
    ang = (math.pi/2.0) - tf.atan(m)   # angle correction
    zmask = tf.where(tf.math.equal(zproj, 0.0), tf.zeros_like(zproj), zmask)
    ang = ang * zmask # place zero where zsum is zero

    ang = ang * z  # weighted by position
    sumz_tot = z * zmask # removing indexes with 0 energies or angles

    #zunmasked = tf.math.reduce_sum(zmask, axis=1) # used for simple mean
    #ang = tf.math.reduce_sum(ang, axis=1)/zunmasked # Mean does not include positions where zsum=0

    ang = tf.math.reduce_sum(ang, axis=1)/tf.math.reduce_sum(sumz_tot, axis=1) # sum ( measured * weights)/sum(weights)
    ang = tf.where(tf.math.equal(amask, 0.), ang, 100. * tf.ones_like(ang)) # Place 100 for measured angle where no energy is deposited in events

    ang = tf.expand_dims(ang, 1)
    return ang


def ecal_angle_keras(image, daxis):         # malfunctioning in TF 2.x
    image = K.squeeze(image, axis=daxis) # remove additional dimension
    
    # get shapes
    x_shape= K.int_shape(image)[1] 
    y_shape= K.int_shape(image)[2]
    z_shape= K.int_shape(image)[3]
    sumtot = K.sum(image, axis=(1, 2, 3))# sum of events

    # get 1. where event sum is 0 and 0 elsewhere
    amask = tf.where(K.equal(sumtot, 0.0), K.ones_like(sumtot) , K.zeros_like(sumtot))  # K.tf.where(K.equal(sumtot, 0.0), K.ones_like(sumtot) , K.zeros_like(sumtot))
    masked_events = K.sum(amask) # counting zero sum events
    
    # ref denotes barycenter as that is our reference point
    x_ref = K.sum(K.sum(image, axis=(2, 3)) * (K.cast(K.expand_dims(K.arange(x_shape), 0), dtype='float32') + 0.5) , axis=1) # sum for x positions * x index
    y_ref = K.sum(K.sum(image, axis=(1, 3)) * (K.cast(K.expand_dims(K.arange(y_shape), 0), dtype='float32') + 0.5), axis=1)
    z_ref = K.sum(K.sum(image, axis=(1, 2)) * (K.cast(K.expand_dims(K.arange(z_shape), 0), dtype='float32') + 0.5), axis=1)
    x_ref = tf.where(K.equal(sumtot, 0.0), K.ones_like(x_ref) , x_ref/sumtot) # return max position if sumtot=0 and divide by sumtot otherwise
    y_ref = tf.where(K.equal(sumtot, 0.0), K.ones_like(y_ref) , y_ref/sumtot)
    z_ref = tf.where(K.equal(sumtot, 0.0), K.ones_like(z_ref), z_ref/sumtot)
    #reshape    
    x_ref = K.expand_dims(x_ref, 1)
    y_ref = K.expand_dims(y_ref, 1)
    z_ref = K.expand_dims(z_ref, 1)

    sumz = K.sum(image, axis =(1, 2)) # sum for x,y planes going along z

    # Get 0 where sum along z is 0 and 1 elsewhere
    zmask = tf.where(K.equal(sumz, 0.0), K.zeros_like(sumz) , K.ones_like(sumz))
        
    x = K.expand_dims(K.arange(x_shape), 0) # x indexes
    x = K.cast(K.expand_dims(x, 2), dtype='float32') + 0.5
    y = K.expand_dims(K.arange(y_shape), 0)# y indexes
    y = K.cast(K.expand_dims(y, 2), dtype='float32') + 0.5
  
    #barycenter for each z position
    x_mid = K.sum(K.sum(image, axis=2) * x, axis=1)
    y_mid = K.sum(K.sum(image, axis=1) * y, axis=1)
    x_mid = tf.where(K.equal(sumz, 0.0), K.zeros_like(sumz), x_mid/sumz) # if sum != 0 then divide by sum
    y_mid = tf.where(K.equal(sumz, 0.0), K.zeros_like(sumz), y_mid/sumz) # if sum != 0 then divide by sum

    #Angle Calculations
    z = (K.cast(K.arange(z_shape), dtype='float32') + 0.5)  * K.ones_like(z_ref) # Make an array of z indexes for all events
    zproj = K.sqrt(K.maximum((x_mid-x_ref)**2.0 + (z - z_ref)**2.0, K.epsilon()))# projection from z axis with stability check
    m = tf.where(K.equal(zproj, 0.0), K.zeros_like(zproj), (y_mid-y_ref)/zproj)# to avoid divide by zero for zproj =0
    m = tf.where(K.less(z, z_ref),  -1 * m, m)# sign inversion
    ang = (math.pi/2.0) - tf.atan(m)# angle correction
    zmask = tf.where(K.equal(zproj, 0.0), K.zeros_like(zproj) , zmask)
    ang = ang * zmask # place zero where zsum is zero
    
    ang = ang * z  # weighted by position
    sumz_tot = z * zmask # removing indexes with 0 energies or angles

    #zunmasked = K.sum(zmask, axis=1) # used for simple mean 
    #ang = K.sum(ang, axis=1)/zunmasked # Mean does not include positions where zsum=0

    ang = K.sum(ang, axis=1)/K.sum(sumz_tot, axis=1) # sum ( measured * weights)/sum(weights)
    ang = tf.where(K.equal(amask, 0.), ang, 100. * K.ones_like(ang)) # Place 100 for measured angle where no energy is deposited in events
    
    ang = K.expand_dims(ang, 1)
    return ang


def disc_loss(generator, discriminator, image_batch, energy_batch, ang_batch, ecal_batch, batch_size, label, \
                    latent_size=256, wtf=6.0, wa=0.2, wang=1, we=0.1, epoch=0):
    # print('image_batch shape: {}'.format(np.shape(image_batch)))
    discriminate = discriminator(image_batch)

    #true/fake loss : binary cross-entropy
    if label == "ones":
        labels = bit_flip_tf(tf.ones_like(discriminate[0])*0.9)     #true=1    
    elif label == "zeros":
        labels = bit_flip_tf(tf.zeros_like(discriminate[0])*0.1)    #fake=0
    
    # nan_tf = np.isnan(discriminate[0])
    # print('Is nan in D_OUT[0]? {}'.format(np.any(nan_tf)))
    loss_true_fake = tf.reduce_mean(- labels * tf.math.log(discriminate[0] + 2e-7) - (1 - labels) * tf.math.log(1 - discriminate[0] + 2e-7))

    # aux loss : MAPE
    # nan_aux = np.isnan(discriminate[1])
    # print('Is nan in D_OUT[1]? {}'.format(np.any(nan_aux)))
    loss_aux = tf.reduce_mean(tf.math.abs((energy_batch - discriminate[1])/(energy_batch + 2e-7))) *100
    
    # ang loss : MAE
    # nan_ang = np.isnan(discriminate[2])
    # print('Is nan in D_OUT[2]? {}'.format(np.any(nan_ang)))
    loss_ang = tf.reduce_mean(tf.math.abs(ang_batch - discriminate[2]))

    # print('Is nan in ecal_batch? {}'.format(np.any(np.isnan(ecal_batch))))
    # print('Is nan in D_OUT[3]? {}'.format(np.any(np.isnan(discriminate[3]))))
    # ecal loss : MAPE
    # nan_ecal = np.isnan(discriminate[3])
    # print('Is nan in D_OUT[3]? {}'.format(np.any(nan_ecal)))
    loss_ecal = tf.reduce_mean(tf.math.abs((ecal_batch - discriminate[3])/(ecal_batch + 2e-7))) *100

    # total loss
    weight_true_fake = wtf
    weight_aux = wa
    weight_ang = wang
    weight_ecal = we
    total_loss = weight_true_fake * loss_true_fake + weight_aux * loss_aux + weight_ang * loss_ang + weight_ecal * loss_ecal
    return total_loss, loss_true_fake, loss_aux, loss_ang, loss_ecal


def gen_loss(generator, discriminator, gen_aux, gen_ang, gen_ecal, batch_size=128, latent_size=256, epoch=10, wtf=6.0, wa=0.2, wang=1, we=0.1):
    noise = np.random.normal(0, 1, (batch_size, latent_size-1))
    # generator_input = np.concatenate((gen_aux.reshape(-1, 1), gen_ang.reshape(-1, 1), noise), axis=1)
    generator_input = np.concatenate((gen_ang.reshape(-1, 1), np.multiply(noise, gen_aux)), axis=1)
    # print('gen_loss: shape of generator input: {}'.format(generator_input.shape))
    # print('gen_loss: any NaNs in generator input: {}'.format(np.any(np.isnan(generator_input))))
    generated_images = generator(generator_input)

    # print('gen_loss: any NaNs in generated images: {}'.format(np.any(np.isnan(np.array(generated_images)))))

    discriminator_fake = discriminator(generated_images)

    # print('gen_loss: any NaNs in discriminator output?')
    # print('tf: {}'.format(np.any(np.isnan(np.array(discriminator_fake[0])))))
    # print('aux: {}'.format(np.any(np.isnan(np.array(discriminator_fake[1])))))
    # print('ang: {}'.format(np.any(np.isnan(np.array(discriminator_fake[2])))))
    # print('ecal: {}'.format(np.any(np.isnan(np.array(discriminator_fake[3])))))
    # print('images in disc: {}'.format(np.any(np.isnan(np.array(discriminator_fake[4])))))
    # print('inv images: {}'.format(np.any(np.isnan(np.array(discriminator_fake[5])))))
    
    #true/fake
    label_fake = bit_flip_tf(tf.ones_like(discriminator_fake[0])*0.9)   #ones = true
    loss_true_fake = tf.reduce_mean(- label_fake * tf.math.log(discriminator_fake[0] + 2e-7) - 
                               (1 - label_fake) * tf.math.log(1 - discriminator_fake[0] + 2e-7))
    # aux : MAPE
    loss_aux = tf.reduce_mean(tf.math.abs((gen_aux - discriminator_fake[1])/(gen_aux + 2e-7))) *100
    # ang loss : MAE
    # temp = tf.math.abs(gen_ang - discriminator_fake[2])
    # print('\nShape of ang abs in gen loss: {}'.format(np.array(temp).shape))
    # loss_ang = tf.reduce_mean(temp)
    loss_ang = tf.reduce_mean(tf.math.abs(gen_ang - discriminator_fake[2]))
    # ecal : MAPE
    # temp = tf.math.abs((gen_ecal - discriminator_fake[3])/(gen_ecal + 2e-7))
    # print('\nShape of ecal abs in gen loss: {}'.format(np.array(temp).shape))
    # loss_ecal = tf.reduce_mean(temp) *100
    loss_ecal = tf.reduce_mean(tf.math.abs((gen_ecal - discriminator_fake[3])/(gen_ecal + 2e-7))) *100
    #total loss
    weight_true_fake = wtf
    weight_aux = wa
    weight_ang = wang
    weight_ecal = we
    total_loss = weight_true_fake * loss_true_fake + weight_aux * loss_aux + weight_ang * loss_ang + weight_ecal * loss_ecal
    # print('gen_loss: is total_loss NaN? {}'.format(np.isnan(total_loss)))
    return [total_loss, loss_true_fake, loss_aux, loss_ang, loss_ecal]


def bit_flip_tf(x, prob = 0.05):
    """ flips a int array's values with some probability """
    x = np.array(x)
    selection = np.random.uniform(0, 1, x.shape) < prob
    x[selection] = 1* np.logical_not(x[selection])
    x = tf.constant(x)
    return x

###########################################################################################################################
# NOT USED FOR THE ANGLE DATA
# 
# # return a fit for Ecalsum/Ep for Ep 
# #https://github.com/svalleco/3Dgan/blob/0c4fb6f7d47aeb54aae369938ac2213a2cc54dc0/keras/EcalEnergyTrain.py#L288
# def GetEcalFit(sampled_energies, particle='Ele', mod=0, xscale=1): #the smaller the energy, the closer is the factor to 2, the bigger the energy, the smaller is the factor
#     if mod==0:
#        return np.multiply(2, sampled_energies)
#     elif mod==1:
#        if particle == 'Ele':
#          root_fit = [0.0018, -0.023, 0.11, -0.28, 2.21]
#          ratio = np.polyval(root_fit, sampled_energies)
#          return np.multiply(ratio, sampled_energies) * xscale
#        elif particle == 'Pi0':
#          root_fit = [0.0085, -0.094, 2.051]
#          ratio = np.polyval(root_fit, sampled_energies)
#          return np.multiply(ratio, sampled_energies) * xscale

        
# def func_for_gen(nb_test, latent_size=200, epoch=0, energ_range = None):
#     noise =            np.random.normal(0, 1, (nb_test, latent_size))  #input for bit_flip() to generate true/false values for discriminator
#     if energ_range is None:
#         if epoch<3:
#             gen_aux =          np.random.uniform(1, 4,(nb_test,1 ))   #generates aux for dicriminator
#         else:
#             gen_aux =          np.random.uniform(0.02, 5,(nb_test,1 ))   #generates aux for dicriminator
#     else:
#         gen_aux =          np.random.uniform(energ_range[0], energ_range[1],(nb_test,1 ))
#     #gen_ecal =         np.multiply(2, gen_aux)                          #generates ecal for discriminator
#     generator_input =  np.multiply(gen_aux, noise)                      #generates input for generator
#     gen_ecal_func =    GetEcalFit(gen_aux, mod=1)
#     return noise, gen_aux, generator_input, gen_ecal_func



# if __name__ == "__main__":
#     discriminator([25,25,25,1], keras_dformat='channels_last').summary()


