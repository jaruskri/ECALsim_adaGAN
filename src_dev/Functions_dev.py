from __future__ import print_function
from collections import defaultdict
from posix import NGROUPS_MAX
try:
    import cPickle as pickle
except ImportError:
    import pickle
import argparse
import os
from six.moves import range
import sys
import h5py 
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  #I need this for my plot

# from loadAngle_ex import FilterData, GetAngleData


#v1_5_8: Some optimizations in the validation functions

def LoadAngleData(datapath, files, file_no, dscale=50., xscale=1., xpower=0.85, yscale=100, angscale=1, angtype='theta', thresh=1e-4, daxis=-1):
    train_file_folder = datapath
    infile = files[file_no]
    dataset = h5py.File(train_file_folder + "//" + infile,'r')
    X = np.array(dataset.get('ECAL'))* xscale
    # X = np.array(dataset.get('ECAL'))
    Y = np.array(dataset.get('energy')) / yscale
    ang = np.array(dataset.get(angtype))
    X[X < thresh] = 0
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    ecal = np.sum(X, axis=(1, 2, 3))

    # print('Ep min {} max {} in load function'.format(np.min(Y), np.max(Y)))
    # print('ECAL min {} max {} in load function'.format(np.min(ecal), np.max(ecal)))

    indexes = np.where(ecal > 10.0)
    X=X[indexes]
    Y=Y[indexes]
    ang=ang[indexes]

    ecal = np.sum(X, axis=(1, 2, 3))
    # print('ECAL min {} max {} in load function'.format(np.min(ecal), np.max(ecal)))

    X = np.expand_dims(X, axis=-1)
    # ecal=ecal[indexes]
    # ecal=np.expand_dims(ecal, axis=daxis)

    # Adjust to get sampling fraction ~ 0.02 ..... only for the image analysis, not for training
    # X = X / dscale

    # To stabilize the training
    if xpower !=1.:
        X = np.power(X, xpower)

    dataset.close()
    return X, Y, ang, ecal # ecal BEFORE using the np.power!

    

def bit_flip(x, prob=0.05):
    """ flips a int array's values with some probability """
    x = np.array(x)
    selection = np.random.uniform(0, 1, x.shape) < prob
    x[selection] = 1 * np.logical_not(x[selection])
    return x


def data_preperation_ang(X, y, ang, ecal, keras_dformat, batch_size, percent=100):      #data preperation
    X[X < 1e-6] = 0  #remove unphysical values

    # print(np.shape(X))

    X_train, X_test, y_train, y_test, ang_train, ang_test, ecal_train, ecal_test = train_test_split(X, y, ang, ecal, train_size=0.9, test_size=0.1)

    #take just a percentage form the data to make fast tests
    X_train=X_train[:int(len(X_train)*percent/100),:]
    y_train=y_train[:int(len(y_train)*percent/100)]
    X_test=X_test[:int(len(X_test)*percent/100),:]
    y_test=y_test[:int(len(y_test)*percent/100)]
    ang_train=ang_train[:int(len(ang_train)*percent/100)]
    ang_test=ang_test[:int(len(ang_test)*percent/100)]
    ecal_train=ecal_train[:int(len(ecal_train)*percent/100)]
    ecal_test=ecal_test[:int(len(ecal_test)*percent/100)]

    y_train= y_train
    y_test=y_test
    """
    print("X_train_shape: ", X_train.shape)
    print("X_test_shape: ", X_test.shape)
    print("y_train_shape: ", y_train.shape)
    print("y_test_shape: ", y_test.shape)
    print('*************************************************************************************')
    """
    #####################################################################################
    nb_train, nb_test = X_train.shape[0], X_test.shape[0]
    if nb_train < batch_size:
        print("\nERROR: batch_size is larger than training data")
        print("batch_size: ", batch_size)
        print("training data: ", nb_train, "\n")

    X_train = X_train.astype(np.float32)  # cast to fp32
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    ang_train = ang_train.astype(np.float32)
    ang_test = ang_test.astype(np.float32)
    ecal_train = ecal_train.astype(np.float32)
    ecal_test = ecal_test.astype(np.float32)

    return X_train, X_test, y_train, y_test, ang_train, ang_test, ecal_train, ecal_test, nb_train, nb_test
        
  
def create_files_list(train_data_path):     #all files have to be within one folder
    train_file_folder = train_data_path
    files = os.listdir(train_file_folder)
    return [fname for fname in files if '.h5' in fname]

def create_folder(folder_name, print_outputs = True):        #create folders in which the trainings progress will be saved
    dirName=folder_name
    try:
        # Create target Directory
        os.mkdir(dirName)
        if print_outputs == True:
            print("Directory " , dirName ,  " Created ") 
    except FileExistsError:
        if print_outputs == True:
            print("Directory " , dirName ,  " already exists")
    return


def plot_loss(train_history, test_history, save_folder, save=False):      #plot the losses as graph
    #generator train loss
    gen_loss=[]
    gen_generation_loss=[]
    gen_auxiliary_loss=[]
    gen_angle_loss = []
    gen_lambda5_loss=[]
    x=[]
    for epoch in range(len(train_history["generator"])):
        x.append(epoch+1)
        gen_loss.append(train_history["generator"][epoch][0])
        gen_generation_loss.append(train_history["generator"][epoch][1])
        gen_auxiliary_loss.append(train_history["generator"][epoch][2])
        gen_angle_loss.append(train_history["generator"][epoch][3])
        gen_lambda5_loss.append(train_history["generator"][epoch][4])

    #generator test loss
    gen_test_loss=[]
    gen_test_generation_loss=[]
    gen_test_auxiliary_loss=[]
    gen_test_angle_loss=[]
    gen_test_lambda5_loss=[]
    for epoch in range(len(test_history["generator"])):
        gen_test_loss.append(test_history["generator"][epoch][0])
        gen_test_generation_loss.append(test_history["generator"][epoch][1])
        gen_test_auxiliary_loss.append(test_history["generator"][epoch][2])
        gen_test_angle_loss.append(test_history["generator"][epoch][3])
        gen_test_lambda5_loss.append(test_history["generator"][epoch][4])


    #discriminator train loss
    disc_loss=[]
    disc_generation_loss=[]
    disc_auxiliary_loss=[]
    disc_angle_loss=[]
    disc_lambda5_loss=[]
    x=[]
    for epoch in range(len(train_history["discriminator"])):
        x.append(epoch+1)
        disc_loss.append(train_history["discriminator"][epoch][0])
        disc_generation_loss.append(train_history["discriminator"][epoch][1])
        disc_auxiliary_loss.append(train_history["discriminator"][epoch][2])
        disc_angle_loss.append(train_history["discriminator"][epoch][3])
        disc_lambda5_loss.append(train_history["discriminator"][epoch][4])

    #discriminator test loss
    disc_test_loss=[]
    disc_test_generation_loss=[]
    disc_test_auxiliary_loss=[]
    disc_test_angle_loss=[]
    disc_test_lambda5_loss=[]
    for epoch in range(len(test_history["discriminator"])):
        disc_test_loss.append(test_history["discriminator"][epoch][0])
        disc_test_generation_loss.append(test_history["discriminator"][epoch][1])
        disc_test_auxiliary_loss.append(test_history["discriminator"][epoch][2])
        disc_test_angle_loss.append(test_history["discriminator"][epoch][3])
        disc_test_lambda5_loss.append(test_history["discriminator"][epoch][4])    

    #generator loss
    plt.title("Total Loss")
    plt.plot(x, gen_test_loss, label = "Generator Test", color ="green", linestyle="dashed")
    plt.plot(x, gen_loss, label = "Generator Train", color ="green")

    #discriminator loss
    plt.plot(x, disc_test_loss, label = "Discriminator Test", color ="red", linestyle="dashed")
    plt.plot(x, disc_loss, label = "Discriminator Train", color ="red")
    
    plt.legend()
    if len(x) >=5:
        plt.ylim(0,40)
    #plt.yscale("log")
    plt.grid("True")
    plt.xlabel('Epoch')
    plt.ylabel('Loss') 
    if save == True:
        plt.savefig(save_folder + "/lossplot.png")    
    plt.show()
    
    ######################################################
    #second plot
    plt.title("Single Losses")
    #plt.plot(x, gen_loss, label = "Generator Total", color ="green")
    plt.plot(x, gen_generation_loss, label = "Gen True/Fake", color ="green")
    plt.plot(x, gen_auxiliary_loss, label = "Gen AUX", color ="lime")
    plt.plot(x, gen_angle_loss, label = "Gen ANG", color ="teal")
    plt.plot(x, gen_lambda5_loss, label = "Gen ECAL", color ="aquamarine")
    
    #plt.plot(x, disc_loss, label = "Discriminator Train", color ="red")
    plt.plot(x, disc_generation_loss, label = "Disc True/Fake", color ="red")
    plt.plot(x, disc_auxiliary_loss, label = "Disc AUX", color ="orange")
    plt.plot(x, disc_angle_loss, label = "Disc ANG", color ="maroon")
    plt.plot(x, disc_lambda5_loss, label = "Disc ECAL", color ="lightsalmon")
    
    plt.legend()
    if len(x) >=5:
        plt.ylim(0,20)
    #plt.yscale("log")
    plt.grid("True")
    plt.xlabel('Epoch')
    plt.ylabel('Loss') 
    if save == True:
        plt.savefig(save_folder + "/single_losses.png")    
    plt.show()
    
    return

#plot for validation metric
def plot_validation(train_history, save_folder):
    validation_loss =[]
    epoch=[]
    for i in range(len(train_history["validation"])):
        epoch.append(i+1)
        validation_loss.append(train_history["validation"][i])
        
    plt.title("Validation Metric")
    plt.plot(epoch, validation_loss, label = "Validation Metric", color ="green")
    plt.legend()
    if len(epoch) >=5:
        plt.ylim(0,1)
    plt.grid("True")
    plt.xlabel('Epoch')
    plt.ylabel('Validation Metric') 
    plt.savefig(save_folder + "/Validation_Plot.png")    
    plt.show()
    return

#gromov-wasserstein-distance
def plot_gromov_w_distance(train_history, save_folder):
    validation_loss =[]
    epoch=[]
    for i in range(len(train_history["Gromov_Wasserstein_validation"])):
        epoch.append(i+1)
        validation_loss.append(train_history["Gromov_Wasserstein_validation"][i])
        
    plt.title("Gromov Wasserstein Distance")
    plt.plot(epoch, validation_loss, label = "Gromov Wasserstein Distance", color ="green")
    plt.legend()
    if len(epoch) >=5:
        plt.ylim(0,0.1)
    plt.grid("True")
    plt.xlabel('Epoch')
    plt.ylabel('Gromov Wasserstein Distance') 
    plt.savefig(save_folder + "/Gromov Wasserstein Distance.png")    
    plt.show()
    return


def loss_table(train_history,test_history, save_folder, epoch = 0, validation_metric = 0,  save=False, timeforepoch=0):        #print the loss table during training
    print('{0:<22s} | {1:4s} | {2:15s} | {3:5s} | {4:5s} | {5:5s}'.format(
            'component', "total_loss", "fake/true_loss", "AUX_loss", "ANG_loss", "ECAL_loss"))
    print('-' * 65)

    ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}| {4:<5.2f}| {4:<5.2f}'
    print(ROW_FMT.format('generator (train)',
                         *train_history['generator'][-1]))
    print(ROW_FMT.format('generator (test)',
                         *test_history['generator'][-1]))
    print(ROW_FMT.format('discriminator (train)',
                         *train_history['discriminator'][-1]))
    print(ROW_FMT.format('discriminator (test)',
                         *test_history['discriminator'][-1]))
    if save == True: 
        if epoch == 0:
            f= open(save_folder + "/loss_table.txt","w")
        else:
            f= open(save_folder + "/loss_table.txt","a")
        str_epoch = "Epoch: " + str(epoch)
        f.write(str_epoch) 
        f.write("\n")
        f.write('{0:<22s} | {1:4s} | {2:15s} | {3:5s}| {4:5s}| {5:5s}'.format('component', "total_loss", "fake/true_loss", "AUX_loss", "ANG_loss", "ECAL_loss"))
        f.write("\n")
        f.write('-' * 65) 
        f.write("\n")
        f.write(ROW_FMT.format('generator (train)', *train_history['generator'][-1])) 
        f.write("\n")
        f.write(ROW_FMT.format('generator (test)', *test_history['generator'][-1]))         
        f.write("\n")
        f.write(ROW_FMT.format('discriminator (train)', *train_history['discriminator'][-1])) 
        f.write("\n")
        f.write(ROW_FMT.format('discriminator (test)', *test_history['discriminator'][-1])) 
        e = timeforepoch
        f.write('\nTime for Epoch: {:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
        f.write("\nValidation Metric: " + str(validation_metric))
        #f.write("\nGromov Wasserstein Distance: " + str(train_history['Gromov_Wasserstein_validation'][-1]))
        f.write("\n\n")
        f.close()                    
    return


def reset_Session():     #a function, which resets the connection to the GPU at the end of the run
    sess = tf.compat.v1.keras.backend.get_session()
    tf.compat.v1.keras.backend.clear_session()
    sess.close()
    sess = tf.compat.v1.keras.backend.get_session()

    try:
        del model # this is from global space - change this as you need
    except:
        pass

    print('---GPU memory reseted') # if it's done something you should see a number being outputted        
        


        
        
        
        
#############################################################################
#functions for gulrukhs validation function

import argparse
import os
from six.moves import range
import sys
import h5py 
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import numpy.core.umath_tests as umath
import math

#functions which can be saved seperately

# get sums along different axis
def get_sums(images):
    sumsx = np.squeeze(np.sum(images, axis=(2,3)))
    sumsy = np.squeeze(np.sum(images, axis=(1,3)))
    sumsz = np.squeeze(np.sum(images, axis=(1,2)))
    return sumsx, sumsy, sumsz

# get moments
def get_moments(sumsx, sumsy, sumsz, totalE, m, x=51, y=51, z=25):
    old_err_state = np.seterr(divide='raise')
    ignored_states = np.seterr(**old_err_state)
    totalE = np.squeeze(totalE)
    index = sumsx.shape[0]
    momentX = np.zeros((index, m))
    momentY = np.zeros((index, m))
    momentZ = np.zeros((index, m))
    ECAL_midX = np.zeros(index)
    ECAL_midY = np.zeros(index)
    ECAL_midZ = np.zeros(index)
    for i in range(m):
      relativeIndices = np.tile(np.arange(x), (index,1))
      moments = np.power((relativeIndices.transpose()-ECAL_midX).transpose(), i+1)
      ECAL_momentX = np.divide(umath.inner1d(sumsx, moments) ,totalE)
      if i==0: ECAL_midX = ECAL_momentX.transpose()
      momentX[:,i] = ECAL_momentX
    for i in range(m):
      relativeIndices = np.tile(np.arange(y), (index,1))
      moments = np.power((relativeIndices.transpose()-ECAL_midY).transpose(), i+1)
      ECAL_momentY = np.divide(umath.inner1d(sumsy, moments), totalE)
      if i==0: ECAL_midY = ECAL_momentY.transpose()
      momentY[:,i]= ECAL_momentY
    for i in range(m):
      relativeIndices = np.tile(np.arange(z), (index,1))
      moments = np.power((relativeIndices.transpose()-ECAL_midZ).transpose(), i+1)
      ECAL_momentZ = np.divide(umath.inner1d(sumsz, moments), totalE)
      if i==0: ECAL_midZ = ECAL_momentZ.transpose()
      momentZ[:,i]= ECAL_momentZ
    return momentX, momentY, momentZ

#Optimization metric
def metric(var, energies, m, angtype='mtheta', x=25, y=25, z=25, ang=1):
    metricp = 0
    metrice = 0
    metrica = 0

    for energy in energies:
        # print('\nValidation - mean of moments')
        #Relative error on mean moment value for each moment and each axis
        x_act= np.mean(var["momentX_act"+ str(energy)], axis=0) # first calculate the average of moment values over the images
        x_gan= np.mean(var["momentX_gan"+ str(energy)], axis=0)
        y_act= np.mean(var["momentY_act"+ str(energy)], axis=0)
        y_gan= np.mean(var["momentY_gan"+ str(energy)], axis=0)
        z_act= np.mean(var["momentZ_act"+ str(energy)], axis=0)
        z_gan= np.mean(var["momentZ_gan"+ str(energy)], axis=0)
        # print('relative moment errors')
        var["posx_error"+ str(energy)]= (x_act - x_gan)/x_act # relative error of the mean moment values
        var["posy_error"+ str(energy)]= (y_act - y_gan)/y_act
        var["posz_error"+ str(energy)]= (z_act - z_gan)/z_act
        #Taking absolute of errors and adding for each axis then scaling by 3
        # Average the moment errors over axes
        # print('average over axis moments')
        var["pos_error"+ str(energy)]= (np.absolute(var["posx_error"+ str(energy)]) + np.absolute(var["posy_error"+ str(energy)])+ np.absolute(var["posz_error"+ str(energy)]))/3
        #Summing over moments and dividing for number of moments
        var["pos_total"+ str(energy)]= np.sum(var["pos_error"+ str(energy)])/m
        metricp += var["pos_total"+ str(energy)]
        
        # Take profile along each axis and find mean along events
        # print('\n Validation - mean of energy profiles')
        sumxact, sumyact, sumzact = np.mean(var["sumsx_act" + str(energy)], axis=0), np.mean(var["sumsy_act" + str(energy)], axis= 0), np.mean(var["sumsz_act" + str(energy)], axis=0)
        sumxgan, sumygan, sumzgan = np.mean(var["sumsx_gan" + str(energy)], axis=0), np.mean(var["sumsy_gan" + str(energy)], axis=0), np.mean(var["sumsz_gan" + str(energy)], axis=0)
        # print('relative error on profiles')
        var["eprofilex_error"+ str(energy)] = np.divide((sumxact - sumxgan), sumxact)
        var["eprofiley_error"+ str(energy)] = np.divide((sumyact - sumygan), sumyact)
        var["eprofilez_error"+ str(energy)] = np.divide((sumzact - sumzgan), sumzact)
        #Take absolute of error and mean for all events
        # print('average error for each profile')
        var["eprofilex_total"+ str(energy)]= np.sum(np.absolute(var["eprofilex_error"+ str(energy)]))/x
        var["eprofiley_total"+ str(energy)]= np.sum(np.absolute(var["eprofiley_error"+ str(energy)]))/y
        var["eprofilez_total"+ str(energy)]= np.sum(np.absolute(var["eprofilez_error"+ str(energy)]))/z

        var["eprofile_total"+ str(energy)]= (var["eprofilex_total"+ str(energy)] + var["eprofiley_total"+ str(energy)] + var["eprofilez_total"+ str(energy)])/3
        metrice += var["eprofile_total"+ str(energy)]
        if ang:
            var["angle_error"+ str(energy)] = np.mean(np.absolute((var[angtype + "_act" + str(energy)] - var[angtype + "_gan" + str(energy)])/var[angtype + "_act" + str(energy)]))
            metrica += var["angle_error"+ str(energy)]
        
    metricp = metricp/len(energies) # averaging over energies
    metrice = metrice/len(energies) # averaging over energies
    if ang:metrica = metrica/len(energies)
    tot = metricp + metrice
    if ang:tot = tot + metrica
    result = [tot, metricp, metrice]
    if ang: result.append(metrica)

    print('Tot: {}, metricp: {}, metrice: {}, metrica: {}'.format(result[0], result[1], result[2], result[3]))
    return result


# Measuring 3D angle from image
def measPython(image): # Working version:p1 and p2 are not used. 3D angle with barycenter as reference point
    image = np.squeeze(image)
    x_shape= image.shape[1]
    y_shape= image.shape[2]
    z_shape= image.shape[3]

    sumtot = np.sum(image, axis=(1, 2, 3))# sum of events
    indexes = np.where(sumtot > 0)
    amask = np.ones_like(sumtot)
    amask[indexes] = 0

    masked_events = np.sum(amask) # counting zero sum events

    x_ref = np.sum(np.sum(image, axis=(2, 3)) * np.expand_dims(np.arange(x_shape) + 0.5, axis=0), axis=1)
    y_ref = np.sum(np.sum(image, axis=(1, 3)) * np.expand_dims(np.arange(y_shape) + 0.5, axis=0), axis=1)
    z_ref = np.sum(np.sum(image, axis=(1, 2)) * np.expand_dims(np.arange(z_shape) + 0.5, axis=0), axis=1)

    x_ref[indexes] = x_ref[indexes]/sumtot[indexes]
    y_ref[indexes] = y_ref[indexes]/sumtot[indexes]
    z_ref[indexes] = z_ref[indexes]/sumtot[indexes]

    sumz = np.sum(image, axis =(1, 2)) # sum for x,y planes going along z

    x = np.expand_dims(np.arange(x_shape) + 0.5, axis=0)
    x = np.expand_dims(x, axis=2)
    y = np.expand_dims(np.arange(y_shape) + 0.5, axis=0)
    y = np.expand_dims(y, axis=2)
    x_mid = np.sum(np.sum(image, axis=2) * x, axis=1)
    y_mid = np.sum(np.sum(image, axis=1) * y, axis=1)
    indexes = np.where(sumz > 0)

    zmask = np.zeros_like(sumz)
    zmask[indexes] = 1
    zunmasked_events = np.sum(zmask, axis=1)

    x_mid[indexes] = x_mid[indexes]/sumz[indexes]
    y_mid[indexes] = y_mid[indexes]/sumz[indexes]
    z = np.arange(z_shape) + 0.5# z indexes
    x_ref = np.expand_dims(x_ref, 1)
    y_ref = np.expand_dims(y_ref, 1)
    z_ref = np.expand_dims(z_ref, 1)

    zproj = np.sqrt((x_mid-x_ref)**2.0  + (z - z_ref)**2.0)
    m = (y_mid-y_ref)/zproj
    z = z * np.ones_like(z_ref)
    indexes = np.where(z<z_ref)
    m[indexes] = -1 * m[indexes]
    ang = (math.pi/2.0) - np.arctan(m)
    ang = ang * zmask

    #ang = np.sum(ang, axis=1)/zunmasked_events #mean
    ang = ang * z # weighted by position
    sumz_tot = z * zmask
    ang = np.sum(ang, axis=1)/np.sum(sumz_tot, axis=1)

    indexes = np.where(amask>0)
    ang[indexes] = 100.
    return ang


# short version of analysis                                                                                                                      
def OptAnalysisShort(var, generated_images, energies, ang=1):
    m=2
    
    x = generated_images.shape[1]
    y = generated_images.shape[2]
    z = generated_images.shape[3]
    for energy in energies:
      if energy==0:
         var["events_gan" + str(energy)]=generated_images
      else:
         var["events_gan" + str(energy)]=generated_images[var["indexes" + str(energy)]]
      #print(var["events_gan" + str(energy)].shape)
    #   print('\nValidation - get_sums:')
      var["ecal_gan"+ str(energy)] = np.sum(var["events_gan" + str(energy)], axis = (1, 2, 3))
      var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)] = get_sums(var["events_act" + str(energy)])
      var["sumsx_gan"+ str(energy)], var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)] = get_sums(var["events_gan" + str(energy)])
    #   print('\nValidation - get_moments:')
      var["momentX_act" + str(energy)], var["momentY_act" + str(energy)], var["momentZ_act" + str(energy)]= get_moments(var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)], var["ecal_act"+ str(energy)], m, x=x, y=y, z=z)
      var["momentX_gan" + str(energy)], var["momentY_gan" + str(energy)], var["momentZ_gan" + str(energy)] = get_moments(var["sumsx_gan"+ str(energy)], var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)], var["ecal_gan"+ str(energy)], m, x=x, y=y, z=z)
    #   print('\nValidation - angle:')
      if ang: var["angle_gan"+ str(energy)]= measPython(var["events_gan" + str(energy)])
    #   print('Calculating metric')
    return metric(var, energies, m, angtype='angle', x=x, y=y, z=z, ang=ang)


def validate_block(generator, files_list, percent=20, keras_dformat='channels_first', data_path="", xscale =1, dscale=50., xpower=0.85, yscale = 100, angscale=1, angtype='theta', thresh=1e-4):
    X=np.zeros((1,51,51,25))
    y=np.zeros((1))
    ang=np.zeros((1))
    print("Getting file to validate...")
    filename = files_list[1]
    print('Validation file 1: {}'.format(filename))
    file = h5py.File(data_path + filename,'r')
    X_file = np.array(file.get('ECAL'))*xscale       # here ECAL refers to the energy depositions in calorimeters (individual "pixels")
    y_file = np.array(file.get('energy'))/yscale     # Ep
    ang_file = np.array(file.get(angtype))
    file.close()

    filename = files_list[2]
    print('Validation file 2: {}'.format(filename))
    file2 = h5py.File(data_path + filename,'r')
    X_file2 = np.array(file2.get('ECAL'))*xscale 
    y_file2 = np.array(file2.get('energy'))/yscale
    ang_file2 = np.array(file2.get(angtype))
    file2.close()

    X = np.concatenate((X_file, X_file2))
    y = np.concatenate((y_file, y_file2))
    ang = np.concatenate((ang_file, ang_file2))

    X = X
    y = y
    ang = ang

    X[X < thresh] = 0  #remove unphysical values

    X = np.delete(X, 0,0)   # remove element 0 from column 0
    y = np.delete(y, 0,0)
    ang = np.delete(ang, 0,0)

    X_val = X
    y_val = y
    ang_val = ang

    X_val=X_val[:int(len(X_val)*percent/100),:]
    y_val=y_val[:int(len(y_val)*percent/100)]
    ang_val=ang_val[:int(len(ang_val)*percent/100)]

    # tensorflow ordering
    X_val = np.expand_dims(X_val, axis=-1)

    if keras_dformat !='channels_last':
        X_val = np.moveaxis(X_val, -1,1)

    nb_val = X_val.shape[0]

    X_val = X_val.astype(np.float32)        # cast to fp32
    y_val = y_val.astype(np.float32)
    ang_val = ang_val.astype(np.float32)
    X_val = np.squeeze(X_val)
    if keras_dformat =='channels_last':
        ecal_val = np.sum(X_val, axis=(1, 2, 3))
    else:
        ecal_val = np.sum(X_val, axis=(2, 3, 4))
    indexes = np.where(ecal_val > 10.0)
    indexes = np.squeeze(indexes)
    X_val=X_val[indexes]
    y_val=y_val[indexes]
    ang_val=ang_val[indexes]
    ecal_val = ecal_val[indexes]

    # Adjust to get sampling fraction ~ 0.02
    X_val = X_val / dscale

    var={}
    tolerance = 5
    # energies = [0, 50, 100, 200, 250, 300, 400, 500]  # for 2-500 GeV dataset
    energies = [120, 140, 160, 180]         # for 100-200 GeV dataset
    data0 = X_val  
    data1 = y_val
    ecal = ecal_val

    # Sorting data by Ep
    print("Recording data...")
    for energy in energies:
        if energy==0:
            var["events_act" + str(energy)]=data0
            var["energy" + str(energy)]=data1
            var["angle_act" + str(energy)]=ang_val
            var["ecal_act" + str(energy)]=ecal
            var["index" + str(energy)] = var["events_act" + str(energy)].shape[0]
        else:
            var["indexes" + str(energy)] = np.where((data1 > (energy - tolerance)/100. ) & ( data1 < (energy + tolerance)/100.))
            var["events_act" + str(energy)]=data0[var["indexes" + str(energy)]]
            var["energy" + str(energy)]=data1[var["indexes" + str(energy)]]
            var["angle_act" + str(energy)]=ang_val[var["indexes" + str(energy)]]
            var["ecal_act" + str(energy)]=ecal[var["indexes" + str(energy)]]
            var["index" + str(energy)] = var["events_act" + str(energy)].shape[0]

    #validation

    # data1 = np.expand_dims(data1, axis=1)
    #var = sortEnergy([np.squeeze(X_test), Y_test], np.squeeze(ecal_test), energies, ang=0)
    nb_test = len(y_val); latent_size =256
    noise = np.random.normal(0.1, 1, (nb_test, latent_size-1))
    generator_input = np.concatenate((ang_val.reshape(-1, 1), np.multiply(noise, data1.reshape(-1,1))), axis=1)     # new version (Florian)
    # generator_input = np.concatenate((data1.reshape(-1, 1), ang_val.reshape(-1, 1), noise), axis=1)       # old version (Gulrukh)

    #sess = tf.compat.v1.Session(graph = infer_graph)
    print("Predicting...")
    generated_images = generator.predict(generator_input,batch_size=128)
    #generated_images = sess.run(l_output, feed_dict = {l_input:generator_ip})
    generated_images= np.squeeze(generated_images)
    # Check for empty images...
    num = np.shape(generated_images)[0]
    print('Number of empty images in the generated dataset: {}'.format(num - np.sum(np.any(generated_images, axis=(1,2,3)))))

    # Reverse the power transformation on generated images
    if xpower !=1.:
        generated_images = np.power(generated_images, 1./xpower)

    generated_images = generated_images / dscale        # scale to 0.02 sampling fraction (just like the MC data)
    result = OptAnalysisShort(var, generated_images, energies, ang=1)
    print('Analysing............')
    # All of the results correspond to mean relative errors on different quantities
    print('Result = ', result[0]) #optimize over result[0]
    return result



############################################################################################
#wasserstein Validation
#Gromov-Wasserstein Validation
#https://github.com/svalleco/3Dgan/blob/Anglegan/keras/misc/GromovWass.py#L287

# def preproc(n, scale=1):
#     return n * scale

# def postproc(n, scale=1):
#     return n/scale

# def load_sorted(sorted_path, energies, ang=0):
#     sorted_files = sorted(glob.glob(sorted_path))
#     srt = {}
#     for f in sorted_files:
#        energy = int(filter(str.isdigit, f)[:-1])
#        if energy in energies:
#           srtfile = h5py.File(f,'r')
#           srt["events_act" + str(energy)] = np.array(srtfile.get('ECAL'))
#           srt["energy" + str(energy)] = np.array(srtfile.get('Target'))
#           if ang:
#              srt["angle" + str(energy)] = np.array(srtfile.get('Angle'))
#           print( "Loaded from file", f)
#     return srt
# import glob
# # sort data for fixed angle
# def get_sorted(datafiles, energies, flag=False, num_events1=10000, num_events2=5000, tolerance=5, thresh=0):
#     srt = {}
#     for index, datafile in enumerate(datafiles):
#         data = GetAngleData(datafile, thresh, num_events1) # What should the num_events be here? GetAngleData for bloc. GetData for cube.
#         X = data[0]
#         sumx = np.sum(np.squeeze(X), axis=(1, 2, 3))
#         indexes= np.where(sumx>0)
#         X=X[indexes]
#         Y = data[1]
#         Y=Y[indexes]
#         for energy in energies:
#             if index== 0:
#                 if energy == 0:
#                     srt["events_act" + str(energy)] = X # More events in random bin
#                     srt["energy" + str(energy)] = Y
#                     if srt["events_act" + str(energy)].shape[0] > num_events1:
#                         srt["events_act" + str(energy)] = srt["events_act" + str(energy)][:num_events1]
#                         srt["energy" + str(energy)] = srt["energy" + str(energy)][:num_events1]
#                         flag=False
#                 else:
#                     indexes = np.where((Y > energy - tolerance ) & ( Y < energy + tolerance))
#                     srt["events_act" + str(energy)] = X[indexes]
#                     srt["energy" + str(energy)] = Y[indexes]
#             else:
#                 if energy == 0:
#                    if flag:
#                     srt["events_act" + str(energy)] = np.append(srt["events_act" + str(energy)], X, axis=0)
#                     srt["energy" + str(energy)] = np.append(srt["energy" + str(energy)], Y, axis=0)
#                     if srt["events_act" + str(energy)].shape[0] > num_events1:
#                         srt["events_act" + str(energy)] = srt["events_act" + str(energy)][:num_events1]
#                         srt["energy" + str(energy)] = srt["energy" + str(energy)][:num_events1]
#                         flag=False
#                 else:
#                     if srt["events_act" + str(energy)].shape[0] < num_events2:
#                         indexes = np.where((Y > energy - tolerance ) & ( Y < energy + tolerance))
#                         srt["events_act" + str(energy)] = np.append(srt["events_act" + str(energy)], X[indexes], axis=0)
#                         srt["energy" + str(energy)] = np.append(srt["energy" + str(energy)], Y[indexes], axis=0)
#                     srt["events_act" + str(energy)] = srt["events_act" + str(energy)][:num_events2]
#                     srt["energy" + str(energy)] = srt["energy" + str(energy)][:num_events2]
#     return srt

# def save_sorted(srt, energies, srtdir, ang=0):
#     safe_mkdir(srtdir)
#     for energy in energies:
#        srtfile = os.path.join(srtdir, "events_{:03d}.h5".format(energy))
#        with h5py.File(srtfile ,'w') as outfile:
#           outfile.create_dataset('ECAL',data=srt["events_act" + str(energy)])
#           outfile.create_dataset('Target',data=srt["energy" + str(energy)])
#           if ang:
#              outfile.create_dataset('Angle',data=srt["angle" + str(energy)])
#        print ("Sorted data saved to {}".format(srtfile))

    
# #Divide files in train and test lists     
# def DivideFiles(FileSearch="/data/LCD/*/*.h5",
#                 Fractions=[.9,.1],datasetnames=["ECAL","HCAL"],Particles=[],MaxFiles=-1):
#     #print ("Searching in :",FileSearch)
#     Files =sorted( glob.glob(FileSearch))
#     #print ("Found {} files. ".format(len(Files)))
#     FileCount=0
#     Samples={}
#     for F in Files:
#         FileCount+=1
#         basename=os.path.basename(F)
#         ParticleName=basename.split("_")[0].replace("Escan","")
#         if ParticleName in Particles:
#             try:
#                 Samples[ParticleName].append(F)
#             except:
#                 Samples[ParticleName]=[(F)]
#         if MaxFiles>0:
#             if FileCount>MaxFiles:
#                 break
#     out=[]
#     for j in range(len(Fractions)):
#         out.append([])
#     SampleI=len(Samples.keys())*[int(0)]
#     for i,SampleName in enumerate(Samples):
#         Sample=Samples[SampleName]
#         NFiles=len(Sample)
#         for j,Frac in enumerate(Fractions):
#             EndI=int(SampleI[i]+ round(NFiles*Frac))
#             out[j]+=Sample[SampleI[i]:EndI]
#             SampleI[i]=EndI
#     return out    

# # get data for fixed angle
# # def GetData(datafile, thresh=0, num_events=5000):
# #    #get data for training
# #     #print( 'Loading Data from .....', datafile)
# #     f=h5py.File(datafile,'r')
# #     y=f.get('energy')[:num_events]
# #     x=np.array(f.get('ECAL')[:num_events])
# #     y=(np.array(y[:]))
# #     if thresh>0:
# #        x[x < thresh] = 0
# #     x = np.expand_dims(x, axis=-1)
# #     x = x.astype(np.float32)
# #     y = y.astype(np.float32)
# #     return x, y

# # generate images
# def generate(g, index, cond, latent=256, concat=1, batch_size=128): # batch_size=50 WHY?
#     energy_labels=np.expand_dims(cond[0], axis=1)
#     if len(cond)> 1: # that means we also have angle
#       angle_labels = cond[1]
#       if concat==1:
#         noise = np.random.normal(0, 1, (index, latent-1))  
#         noise = energy_labels * noise
#         gen_in = np.concatenate((angle_labels.reshape(-1, 1), noise), axis=1)
#       elif concat==2:
#         noise = np.random.normal(0, 1, (index, latent-2))
#         gen_in = np.concatenate((energy_labels, angle_labels.reshape(-1, 1), noise), axis=1)
#       else:  
#         noise = np.random.normal(0, 1, (index, 2, latent))
#         angle_labels=np.expand_dims(angle_labels, axis=1)
#         gen_in = np.concatenate((energy_labels, angle_labels), axis=1)
#         gen_in = np.expand_dims(gen_in, axis=2)
#         gen_in = gen_in * noise
#     else:
#       noise = np.random.normal(0, 1, (index, latent))
#       #energy_labels=np.expand_dims(energy_labels, axis=1)
#       gen_in = energy_labels * noise
#     generated_images = g.predict(gen_in, verbose=False, batch_size=batch_size)
#     return generated_images

# import scipy as sp
# # import ot
# def Gromov_metric(var, energies, m, angtype='mtheta', x=25, y=25, z=25, ang=1):
#    metricp = 0
#    metrice = 0
#    metrica = 0
#    metrics = 0
#    for i, energy in enumerate([energies[0]]):
#      if i==0:
#          moment_act=np.hstack((var["momentX_act" + str(energy)], var["momentY_act" + str(energy)], var["momentZ_act" + str(energy)]))
#          shapes_act = np.hstack((var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)]))
#          if ang: angles_act= np.reshape(var["mtheta_act" + str(energy)], (-1, 1))
         
#          #print(var["sf_act"+ str(energy)].shape)
#          sampfr_act = np.reshape(var["sf_act"+ str(energy)], (-1, 1))
#          #sampfr_act = var["sf_act"+ str(energy)]
#          moment_gan=np.hstack((var["momentX_gan" + str(energy)], var["momentY_gan" + str(energy)], var["momentZ_gan" + str(energy)]))
#          shapes_gan = np.hstack((var["sumsx_gan"+ str(energy)], var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)]))
#          if ang: angles_gan= np.reshape(var["mtheta_gan" + str(energy)], (-1, 1))
#          #print(var["sf_gan"+ str(energy)].shape)
#          sampfr_gan = np.reshape(var["sf_gan"+ str(energy)], (-1, 1))
#          #sampfr_gan = var["sf_gan"+ str(energy)]
#          #print(sampfr_gan.shape)
         
#      #print(var)
#      #print(energies)
#      #print(moment_act.shape)
#      #print(shapes_act.shape)
#      #print(sampfr_act.shape)
#      #print(moment_gan.shape)
#      #print(shapes_gan.shape)
#      #print(sampfr_gan.shape)
#      metric_act = np.hstack((moment_act, shapes_act, sampfr_act))
#      metric_gan = np.hstack((moment_gan, shapes_gan, sampfr_gan))
#      #print("gan shape ", metric_gan.shape)
#      #print("act shape ", metric_act.shape)

#      metric_a = np.transpose(metric_act)
#      metric_g = np.transpose(metric_gan)
#      #print("a shape ", metric_a.shape)
#      #print("g shape ", metric_g.shape)
     
#      a = (0.25 /127.)* np.ones(126)  # Changed to 126 from 74.
#      b = (0.25/6.) *np.ones(6)
#      c = 0.25 *np.ones(2) # Changed to 3 from 2. 
    
#      p = np.concatenate((a, b, c))
#      q = np.concatenate((a, b, c))
#      #print("p shape ", p.shape)
#      #print("q shape ", q.shape)
#      C1 = sp.spatial.distance.cdist(metric_a, metric_a, 'correlation')
#      #print("c1 shape ", C1.shape)
#      C2 = sp.spatial.distance.cdist(metric_g, metric_g, 'correlation')
#      #print("c2 shape ", C2.shape)
#      C1 = C1/np.amax(C1)
#      C2 = C2/np.amax(C2)
#      #gw0, log0 = ot.gromov.gromov_wasserstein(C1, C2, p, q, 'square_loss', verbose=True, log=True)
#      #gw, log = ot.gromov.entropic_gromov_wasserstein(C1, C2, p, q, 'kl_loss', epsilon=5e-2, log=True, verbose=True)
#      #print("gw")
#      gw, log = ot.gromov.entropic_gromov_wasserstein(C1, C2, p, q, 'square_loss', epsilon=5e-2, log=True, verbose=True)
#      print('Gromov-Wasserstein distances: ' + str(log['gw_dist']))
#    return log['gw_dist']

