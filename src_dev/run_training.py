##################
#   Example of the run command:
#       python run_training.py --nbfiles 5 --nb_epochs 4
##################

from __future__ import print_function
from cmath import isnan

import os
import numpy as np
import matplotlib.pyplot as plt
import time
import math as math
import argparse

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
import pickle
from six.moves import range

import tensorflow as tf
print(tf.__version__)

# GPU = 1      # Or 0,1, 2, 3, etc.
# os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)  
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

####################################################################################################

# storage = "eos" # "eos", "local"

# Imports of architectures nad other functions
from Models_dev import *
from Functions_dev import *

####################################################################################################

def get_parser():
    # To parse the input parameters to the script
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuno', type=int, default=0, help='which gpu to run on')
    parser.add_argument('--dataset', type=str, default='reduced_Ep', help='reduced_Ep dataset (100-200 GeV) or full_Ep dataset (2-500 GeV)')
    parser.add_argument('--outdir', type=str, default='test', help='path to save the outputs')
    parser.add_argument('--nbfiles', type=int, default=0, help='number of data files to use for training')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--lrate_g', type=float, default=0.0001, help='generator learning rate')
    parser.add_argument('--lrate_d', type=float, default=0.0001, help='discriminator learning rate')
    parser.add_argument('--nb_epochs', type=int, default=30, help='number of training epochs')
    return parser


def main():
    parser = get_parser()
    params = parser.parse_args()

    ### GPUs ###

    gpus = tf.config.list_physical_devices('GPU')
    gpu_id = params.gpuno

    if gpus:
    # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
            tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    # tf.debugging.enable_check_numerics()

    ####################################################################################################

    argsdict = {
        "keras_dformat" : 'channels_last',
        "dshape" : (51,51,25,1),
        "daxis" : (1,2,3),
        "axis" : -1,
        "nb_epochs" : params.nb_epochs, 
        "percent" : 100,                     # % of data to use for a trial run
        "nb_train_files" : params.nbfiles,           # 28 files for 100-200 GeV (~5000 samples each), 80 files for 2-500 GeV (~5000 samples each) but only 40 used for optuna (to make things faster)
                                            # if 0, then the number of files is detected automaticaly
        "lrate_g" : params.lrate_g,
        "lrate_d" : params.lrate_d,
        "save_only_best_weights" : False,
        "latent_size" : 256,
        "batch_size" :  params.batch_size,                  #batch_size must be less or equal test size, otherwise error
        "wtf" : 3.0, #6.0                           # weight true-fake loss
        "wa" : 0.1, #0.2                            # weight auxiliary loss (lin. reg. on Ep)
        "wang" : 25,                                # angle loss
        "we" : 0.1, #0.1                            # ecal loss
        "save_folder" : params.outdir
    }

    if params.dataset == 'reduced_Ep':
        # 100-200 GeV, 28 files, 5000 samples each
        argsdict["train_file_by_file_path"] = "/eos/home-k/kjarusko/Projects/DataAngle/h5files/h5files/"
    elif params.dataset == 'full_Ep':
        # 2-500 GeV, 80 files, 5000 samples each
        argsdict["train_file_by_file_path"] = "/eos/home-k/kjarusko/Projects/DataAngle/Ele_RandomAngle/Ele_RandomAngle/"
    else:
        raise ValueError('Invalid --dataset specification -> choose reduced_Ep of full_Ep.')


    #create output folders for saving files and weights
    print(argsdict['save_folder'])
    create_folder(argsdict['save_folder']+'/')
    create_folder(argsdict['save_folder']+'/Weights/')
    create_folder(argsdict['save_folder']+'/Weights/disc')
    create_folder(argsdict['save_folder']+'/Weights/gen')
    print('*************************************************************************************')

    files = create_files_list(argsdict['train_file_by_file_path']) # returns a list of .h5 files
    print('Files: {}'.format(files))
    argsdict["files"] = files

    if argsdict['nb_train_files'] == 0:
        argsdict['nb_train_files'] = len(files)

    print('\nNumber of training files: {} \n'.format(argsdict['nb_train_files']))

    ########################################################################################################

    # lrate decay function - exponential decay
    def l_dec(initial_lrate, epoch, start_decay=80, decay_rate=0.012):
        epoch = epoch - 1 #because training starts at epoch 1
        if epoch < start_decay:
            k = 0.0
        else:
            k = decay_rate
            epoch = epoch - start_decay
        lrate = initial_lrate * math.exp(-k*epoch)
        return lrate
    
    
    lrate_g = argsdict['lrate_g']
    lrate_d = argsdict['lrate_d']
    print('\n\nStarting lrate_g: {}'.format(lrate_g))
    print('Starting lrate_d: {}'.format(lrate_d))

    X_test = []
    y_test = []
    ang_test = []
    ecal_test = []

    # initialize the architectures - no LeakyReLU in the last layer ! (because of inverting the power transform. on images)
    discriminator = discriminator_kj(keras_dformat = argsdict["keras_dformat"])
    generator = generator_kj(latent_size = argsdict['latent_size'], keras_dformat = argsdict["keras_dformat"], leaky_last_step = False)
    start_epoch = 1
    epoch=0
    # discriminator.summary()
    # generator.summary()

    verbose = 'false'

    train_history = defaultdict(list)   #create a dict with an empty list 
    test_history = defaultdict(list)

    # Start training from epoch 1
    for epoch in range(start_epoch, argsdict['nb_epochs'] + 1):
        print('Epoch {} of {}'.format(epoch, argsdict['nb_epochs']))
        start_epoch = time.time()
        lr_d = l_dec(lrate_d, epoch)
        lr_g = l_dec(lrate_g, epoch)
        optimizer_d = tf.optimizers.Adam(lr_d)
        optimizer_g = tf.optimizers.Adam(lr_g)
        epoch_gen_loss = []
        epoch_disc_loss = []

        if 'X_test' in locals():
            del X_test
            del y_test
            del ang_test
            del ecal_test

        #Iterate over the number of training files
        for file_number in range(argsdict['nb_train_files']):

            #import data file by file method
            start_import = time.time()
            print("File: ", file_number+1, "/", argsdict['nb_train_files'])
            X, y, ang, ecal = LoadAngleData(argsdict['train_file_by_file_path'], argsdict['files'], file_number)
            X_train, X_test_file, y_train, y_test_file, ang_train, ang_test_file, ecal_train, ecal_test_file, nb_train, nb_test = data_preperation_ang(X, y, ang, ecal, \
                                                                                    argsdict['keras_dformat'], argsdict['batch_size'], argsdict['percent'])

            y_train = np.expand_dims(y_train, axis=-1)  # [1,2,3]->[[1],[2],[3]]
            y_test_file = np.expand_dims(y_test_file, axis=-1)
            ang_train = np.expand_dims(ang_train, axis=-1)
            ang_test_file = np.expand_dims(ang_test_file, axis=-1)
            ecal_train = np.expand_dims(ecal_train, axis=-1)
            ecal_test_file = np.expand_dims(ecal_test_file, axis=-1)
            if 'X_test' in locals():
                X_test = np.concatenate((X_test, X_test_file))
                y_test = np.concatenate((y_test, y_test_file))
                ang_test = np.concatenate((ang_test, ang_test_file))
                ecal_test = np.concatenate((ecal_test, ecal_test_file))
            else: 
                X_test = X_test_file
                y_test = y_test_file
                ang_test = ang_test_file
                ecal_test = ecal_test_file
            nb_test=len(y_test)
            end_import = time.time()
            e2 = int(end_import-start_import)
            # print('Time for Import: {:02d}:{:02d}:{:02d}'.format(e2 // 3600, (e2 % 3600 // 60), e2 % 60))

            rounding = math.floor(file_number/2)   # adjusting the learning rate decay - because there are only 5.000 samples in a file, not 10.000 like in the E_p-only data 
            lr_d = l_dec(lrate_d, epoch*20+rounding-20)
            lr_g = l_dec(lrate_g, epoch*20+rounding-20)
            optimizer_d = tf.optimizers.Adam(lr_d)
            optimizer_g = tf.optimizers.Adam(lr_g)
            print("Learnrate Generator:     ", lr_g)
            print("Learnrate Discriminator: ", lr_d)
            nb_batches = int(X_train.shape[0] / argsdict['batch_size'])
            if verbose:
                progress_bar = tf.keras.utils.Progbar(target=nb_batches)
            ################################################################################

            nonzeroim = list()

            batch_size = argsdict["batch_size"]
            #training; loop over epochs
            for batch in range(nb_batches):
                # print('Batch {}'.format(batch))
                # if verbose:
                #     progress_bar.update(batch+1)
                # else:
                #     if epoch % 100 == 0:
                #         print('processed {}/{} batches'.format(batch + 1, nb_batches))

                #create batches
                image_batch  = X_train[(batch*batch_size) : ((batch+1)*batch_size)]
                energy_batch = y_train[(batch*batch_size) : ((batch+1)*batch_size)]
                ang_batch = ang_train[(batch*batch_size) : ((batch+1)*batch_size)]
                ecal_batch   = ecal_train[(batch*batch_size) : ((batch+1)*batch_size)]

                #discriminator true training
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(discriminator.trainable_variables)
                    d_loss_true = disc_loss(generator, discriminator, image_batch, energy_batch, ang_batch, ecal_batch, batch_size, label ="ones", wtf=argsdict['wtf'], wa=argsdict['wa'], wang=argsdict['wang'], we=argsdict['we'], epoch=epoch)
                    temp = np.any(np.isnan(d_loss_true))
                    # print('NaN in true DISC training: {}'.format(temp))
                    d_grads = tape.gradient( d_loss_true[0] , discriminator.trainable_variables )
                optimizer_d.apply_gradients( zip( d_grads , discriminator.trainable_variables) )
                if temp == True:
                    print('Batch {}'.format(batch))
                    print(np.array(d_loss_true))
                    print('NaN loss in true DISC training.')
                    return

                #discriminator fake training
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(discriminator.trainable_variables)
                    noise = np.random.normal(0, 1, (batch_size, argsdict['latent_size']-1))
                    gen_aux = energy_batch
                    gen_ang = ang_batch
                    gen_ecal = ecal_batch
                    generator_input = np.concatenate((gen_ang.reshape(-1, 1), np.multiply(noise,gen_aux)), axis=1)
                    generated_images = generator(generator_input)
                    np_gen_images = np.array(generated_images)
                    
                    # Check for empty images (0s only)
                    nonzero_flag = np.any(np_gen_images, axis=(1,2,3,4))   # 1 if nonzero elements, 0 only if all pixels are 0
                    nonzeroim.append(np.sum(nonzero_flag))
                    d_loss_fake = disc_loss(generator, discriminator, generated_images, gen_aux, gen_ang, gen_ecal, batch_size, label = "zeros", wtf=argsdict['wtf'], wa=argsdict['wa'], wang=argsdict['wang'], we=argsdict['we'], epoch=epoch)
                    temp = np.any(np.isnan(d_loss_fake))
                    d_grads = tape.gradient( d_loss_fake[0] , discriminator.trainable_variables )
                optimizer_d.apply_gradients( zip( d_grads , discriminator.trainable_variables) )
                if temp == True:
                    print('Batch {}'.format(batch))
                    print(np.array(d_loss_fake))
                    print('NaN loss in fake DISC training.')
                    return

                # Average of true and fake disc loss
                d_loss = []
                d_loss.append( (d_loss_true[0] + d_loss_fake[0])/2) # total loss
                d_loss.append( (d_loss_true[1] + d_loss_fake[1])/2) # tru/fake loss
                d_loss.append( (d_loss_true[2] + d_loss_fake[2])/2) # aux loss
                d_loss.append( (d_loss_true[3] + d_loss_fake[3])/2) # ang loss
                d_loss.append( (d_loss_true[4] + d_loss_fake[4])/2) # ecal loss
                epoch_disc_loss.append([d_loss[0].numpy(), d_loss[1].numpy(), d_loss[2].numpy(), d_loss[3].numpy(), d_loss[4].numpy()])  

                #generator training
                gen_losses = []
                for i in range(1): # 1 originaly
                    with tf.GradientTape(watch_accessed_variables=False) as tape:
                        tape.watch(generator.trainable_variables)
                        g_loss = gen_loss(generator, discriminator, gen_aux, gen_ang, gen_ecal, batch_size=batch_size, epoch=epoch, wtf=argsdict['wtf'], wa=argsdict['wa'], wang=argsdict['wang'], we=argsdict['we'])
                        temp = np.any(np.isnan(g_loss))
                        g_grads = tape.gradient( g_loss[0] , generator.trainable_variables )
                    optimizer_g.apply_gradients( zip( g_grads , generator.trainable_variables ) )
                    gen_losses.append([g_loss[0].numpy(), g_loss[1].numpy(), g_loss[2].numpy(), g_loss[3].numpy(), g_loss[4].numpy()])
                epoch_gen_loss.append(np.mean(gen_losses, axis = 0))
                if temp == True:
                    print('Batch {}'.format(batch))
                    print(np.array(gen_losses))
                    print('NaN loss in GEN training.')
                    return
            if sum(nonzeroim) == 0:
                print('All generated images were empty - killing training.')
                return

        ################################################################################
        #testing
        nb_batches = int(X_test.shape[0] / batch_size)
        if nb_batches == 0:  #I need this or I get an error if I reduce the data with percentage
            nb_batches = 1
        disc_test_loss_list = []
        gen_test_loss_list = []
        for batch in range(nb_batches):
            #create batches
            image_batch  = X_test[(batch*batch_size) : ((batch+1)*batch_size)]
            energy_batch = y_test[(batch*batch_size) : ((batch+1)*batch_size)]
            ang_batch = ang_test[(batch*batch_size) : ((batch+1)*batch_size)]
            ecal_batch   = ecal_test[(batch*batch_size) : ((batch+1)*batch_size)]

            d_test_loss_true = disc_loss(generator, discriminator, image_batch, energy_batch, ang_batch, ecal_batch, batch_size, label ="ones", wtf=argsdict['wtf'], wa=argsdict['wa'], wang=argsdict['wang'], we=argsdict['we'])
            
            noise = np.random.normal(0, 1, (batch_size, argsdict['latent_size']-1))
            gen_aux = energy_batch
            gen_ang = ang_batch
            gen_ecal = ecal_batch
            # generator_input = np.concatenate((gen_aux.reshape(-1, 1), gen_ang.reshape(-1, 1), noise), axis=1)   # old version of input (Gulrukh)
            generator_input = np.concatenate((gen_ang.reshape(-1, 1), np.multiply(noise, gen_aux)), axis=1)       # new version of input - Ep*noise (Florian)
            generated_images = generator(generator_input)
            d_test_loss_fake = disc_loss(generator, discriminator, generated_images, gen_aux, gen_ang, gen_ecal, batch_size, label = "zeros", wtf=argsdict['wtf'], wa=argsdict['wa'], wang=argsdict['wang'], we=argsdict['we'])
            d_test_loss = []
            d_test_loss.append( (d_test_loss_true[0] + d_test_loss_fake[0])/2)
            d_test_loss.append( (d_test_loss_true[1] + d_test_loss_fake[1])/2)
            d_test_loss.append( (d_test_loss_true[2] + d_test_loss_fake[2])/2)
            d_test_loss.append( (d_test_loss_true[3] + d_test_loss_fake[3])/2)
            d_test_loss.append( (d_test_loss_true[3] + d_test_loss_fake[4])/2)
            disc_test_loss_list.append([d_test_loss[0].numpy(), d_test_loss[1].numpy(), d_test_loss[2].numpy(), d_test_loss[3].numpy(), d_test_loss[4].numpy()])

            gen_test_loss = gen_loss(generator, discriminator, gen_aux, gen_ang, gen_ecal, batch_size, epoch=epoch, wtf=argsdict['wtf'], wa=argsdict['wa'], wang=argsdict['wang'], we=argsdict['we'])
            gen_test_loss_list.append([gen_test_loss[0].numpy(), gen_test_loss[1].numpy(), gen_test_loss[2].numpy(), gen_test_loss[3].numpy(), gen_test_loss[4].numpy()])

        ###############################
        #validation script
        validation_metric = validate_block(generator, argsdict['files'], percent=argsdict['percent'], keras_dformat=argsdict['keras_dformat'], data_path=argsdict['train_file_by_file_path'])
        ##############################
        #loss dict
        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)   #mean disc loss for all epochs
        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)
        discriminator_test_loss = np.mean(np.array(disc_test_loss_list), axis=0)
        generator_test_loss = np.mean(np.array(gen_test_loss_list), axis=0)

        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)
        train_history['validation'].append(validation_metric)
        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)
        ##############################
        #calculate time for epoch
        end_batch = time.time()
        e = int(end_batch-start_epoch)
        print('Time for Epoch: {:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))

        #save history

        #print loss table and plot generated image; also save them
        loss_table(train_history, test_history, argsdict['save_folder'], epoch, validation_metric[0], save=True, timeforepoch = e)
        
        # save history dict and weights
        pkl_name = 'training_hist_g_{}_d_{}.pkl'.format(lrate_g, lrate_d)
        pickle.dump([train_history, test_history], open(os.path.join(argsdict['save_folder'], pkl_name), 'wb'))
        generator.save_weights(argsdict['save_folder']+"/Weights/gen/params_generator_epoch_"+str(epoch)+".hdf5",overwrite=True)
        discriminator.save_weights(argsdict['save_folder']+"/Weights/disc/params_discriminator_epoch_"+str(epoch)+".hdf5",overwrite=True)
        
        if isnan(validation_metric[0]):
            print('NaN validation - killing training.')
            return
        
    vals = []
    for e in range(len(train_history['validation'])):
            vals.append(train_history['validation'][e][0])
    best_val = np.min(vals)
    best_ind = vals.index(best_val)+1
    print('\nTraining successfull.')
    print('Starting lrate_g: {}'.format(lrate_g))
    print('Starting lrate_d: {}'.format(lrate_d))
    print("Best Weights Epoch: ", best_ind)
    print('Best validation value: {}'.format(best_val))


if __name__ == '__main__':
    main()



