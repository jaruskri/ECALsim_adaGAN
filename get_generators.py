import os
import sys
import numpy as np
import h5py

from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle
    
from adagan_v6 import *

#############
#
# Run training for 2 epochs and save generators in .pb files
#
#############

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


results_dir = '/eos/home-k/kjarusko/Projects/ensemGAN/3Dgan/adaGAN_gpu_v6/run_g10_jan2/'
num_gens = 10
num_epochs = 30

save_dir = os.path.join(results_dir, 'IntelShare')
create_folder(save_dir)
create_folder(os.path.join(save_dir, 'Generators'))
create_folder(os.path.join(save_dir, 'Discriminators'))

path = results_dir + 'Info/'
filename = 'adagan_info.pkl'
file = open(path+filename, 'rb')
file_data = pickle.load(file)
file.close()

params = file_data['params'][0]
params['steps_made'] = 10

gan_bestval = file_data['gan_bestval']

# params['data_dir'] = '/eos/home-k/kjarusko/Projects/DataEnerg/unzipped/'

ada = Adagan(params)

# step = 1
for step in range(0, params['steps_made']):
    create_folder(os.path.join(results_dir, 'IntelShare', 'Generators', 'gen_{}'.format(step)))
    start_epoch = ada._gens_bestepoch[step]
    gen = Generator(params, arch_overwrite="FloG1")
    gen.load_w_epoch(params, step, ada._gens_bestepoch[step])
    # gen.gen.summary()

    disc = Discriminator(params)
    disc.load_w_epoch(params, step, ada._gens_bestepoch[step])
    
    if step == 0:
        data_weights = None
        indices_train_list = None
        indices_test_list = None
    else:
        path = os.path.join(results_dir, 'DataWeights/dataw_step_{}.pkl'.format(step))
        file = open(path, 'rb')
        data_weights = pickle.load(file)
        file.close()

        path = os.path.join(results_dir, 'DataWeights/indices_step_{}.pkl'.format(step))
        file = open(path, 'rb')
        indices = pickle.load(file)
        file.close()
        
        nb_files = len(ada.data_obj.filenames)
        full = [i for i in range(10000)]
        print(np.shape(full))
        indices_test_list = []
        for file_no in range(nb_files):
            temp = indices[file_no]
            indices_test_list.append(list(set(full)-set(temp)))
        indices_train_list = indices
    
    start_epoch = ada._gens_bestepoch[step]
    
    train_history = defaultdict(list)   #create a dict with an empty list 
    test_history = defaultdict(list)
    for epoch in range(start_epoch+1, start_epoch+3): # start from epoch 1
        print('\nEpoch {} of {}'.format(epoch, params['nb_epochs']))
        tick_epoch = time.time()

        # loss lists
        epoch_gen_loss = []
        epoch_disc_loss = []

        if 'X_test' in locals():
            del X_test
            del Ep_test
            del ecal_test

        nb_files = len(ada.data_obj.filenames)
        print('nb_files : {}'.format(nb_files))
        for file_no in range(nb_files):
            tick_import = time.time()
            print("File: ", file_no+1, "/", nb_files)
            ada.data_obj.load_file(params, file_no)
            if data_weights is None:
                X_train, X_test_file, Ep_train, Ep_test_file, ecal_train, ecal_test_file = ada.data_obj.data_preparation(params)
            else:
                dw_train = data_weights[file_no]
                ind_train = indices_train_list[file_no]
                ind_test = indices_test_list[file_no]
                X_train, X_test_file, Ep_train, Ep_test_file, ecal_train, ecal_test_file = ada.data_obj.data_preparation(params, dw_train, ind_train, ind_test)
                # dw_train = dw_train/np.sum(dw_train)
                # dw_test = dw_test/np.sum(dw_test)
            if 'X_test' in locals():
                X_test = np.concatenate((X_test, X_test_file))
                Ep_test = np.concatenate((Ep_test, Ep_test_file))
                ecal_test = np.concatenate((ecal_test, ecal_test_file))
            else: 
                X_test = X_test_file
                Ep_test = Ep_test_file
                ecal_test = ecal_test_file
            tock_import = time.time()
            t_import = int(tock_import-tick_import)
            print('Time for Import: {:02d}:{:02d}:{:02d}'.format(t_import // 3600, (t_import % 3600 // 60), t_import % 60))
            ada.data_obj.empty_datavars()
            train_size = Ep_train.shape[0]
            test_size = Ep_test.shape[0]
            # print(train_size)
            # print(test_size)

            # learning rates
            # print('params[lrate_d]: {}'.format(params['lrate_d']))
            # print('params[lrate_g]: {}'.format(params['lrate_g']))
            lr_d = l_dec(params['lrate_d'], epoch*20+file_no-20) # it decays depending on the epoch and the number of the file
            lr_g = l_dec(params['lrate_g'], epoch*20+file_no-20)
            # # optimizers
            optimizer_d = tf.optimizers.Adam(lr_d)
            optimizer_g = tf.optimizers.Adam(lr_g)
            print("Learnrate Gen:   ", lr_g)
            print("Learnrate Disc:   ", lr_d)

            # Get number of batches
            nb_batches = int(X_train.shape[0] / params['batch_size'])
            if params['verbose']:
                progress_bar = tf.keras.utils.Progbar(target=nb_batches)

            # TRAINING IN BATCHES
            print("GAN training - number of batches: {}".format(nb_batches))
            for batch_no in range(nb_batches):
                if params['verbose']:
                    progress_bar.update(batch_no+1)
                else:
                    if epoch % 100 == 0:
                        print('processed {}/{} batches'.format(batch_no + 1, nb_batches))

                # Create train batches
                batch_size = params['batch_size']
                if data_weights is None:
                    image_batch  = X_train[(batch_no*batch_size) : ((batch_no+1)*batch_size)]
                    energy_batch = Ep_train[(batch_no*batch_size) : ((batch_no+1)*batch_size)]
                    ecal_batch   = ecal_train[(batch_no*batch_size) : ((batch_no+1)*batch_size)]
                else:
                    # Sampling batch using data_weights
                    dw_train = dw_train/np.sum(dw_train)
                    data_ids = np.random.choice(train_size, batch_size, replace=True, p=dw_train)
                    image_batch = X_train[data_ids]
                    energy_batch = Ep_train[data_ids]
                    ecal_batch = ecal_train[data_ids]

                # noise, gen_aux, generator_input, gen_ecal = self._func_for_gen(nb_test=batch_size, epoch=epoch) 
                # generated_images = self.gen(generator_input)
                generated_images, gen_aux, gen_ecal = gen.generate(num=batch_size, epoch=epoch)

                d_loss = disc.train_step(params, optimizer_d, image_batch, energy_batch, ecal_batch, \
                                                generated_images, gen_aux, gen_ecal)
                epoch_disc_loss.append(d_loss)  

                g_loss = gen.train_step(params, optimizer_g, batch_size, epoch, disc)

                epoch_gen_loss.append(g_loss)


        # TESTING IN BATCHES - on the last dataset only
        nb_batches = int(X_test.shape[0] / batch_size)
        if nb_batches == 0:  #I need this or I get an error if I reduce the data with percentage
            nb_batches = 1
        # Prepare list for saving losses
        disc_test_loss_list = []
        gen_test_loss_list = []
        for batch_no in range(nb_batches):
            #create batches
            image_batch  = X_test[(batch_no*batch_size) : ((batch_no+1)*batch_size)]
            energy_batch = Ep_test[(batch_no*batch_size) : ((batch_no+1)*batch_size)]
            ecal_batch   = ecal_test[(batch_no*batch_size) : ((batch_no+1)*batch_size)]

            # Get test loss
            generated_images, gen_aux, gen_ecal = gen.generate(num=batch_size, epoch=epoch)

            d_test_loss = disc.test_step(params, image_batch, energy_batch, ecal_batch, \
                                                generated_images, gen_aux, gen_ecal)
            disc_test_loss_list.append(d_test_loss)

            gen_test_loss = gen.test_step(params, batch_size, epoch, disc)
            gen_test_loss_list.append(gen_test_loss)

        # print('epoch_disc_loss shape: {}'.format(np.shape(epoch_disc_loss)))
        # print('epoch_gen_loss shape: {}'.format(np.shape(epoch_gen_loss)))
        # print('disc_test_loss_list shape: {}'.format(np.shape(disc_test_loss_list)))
        # print('gen_test_loss_list shape: {}'.format(np.shape(gen_test_loss_list)))

        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)   # average losses over batches (average losses of the given epoch)
        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)
        discriminator_test_loss = np.mean(np.array(disc_test_loss_list), axis=0)
        generator_test_loss = np.mean(np.array(gen_test_loss_list), axis=0)

        tock_epoch = time.time()
        t_epoch = int(tock_epoch-tick_epoch)
        print('Time for Epoch: {:02d}:{:02d}:{:02d}'.format(t_epoch // 3600, (t_epoch % 3600 // 60), t_epoch % 60))

        if data_weights is None:
            validation_metric = validate(gen, percent=params['percent'], keras_dformat=params['keras_dformat'], data_path=params['data_dir'])
        else:
            validation_metric = validate_weighted(gen, params['data_dir'], ada.data_obj.filenames, data_weights, indices_train_list, \
                                                  params['percent'], params['keras_dformat'])
            
        # Append epoch losses to dictionary
        train_history['generator'].append(generator_train_loss) # [total, True/Fake, aux, ecal]
        train_history['discriminator'].append(discriminator_train_loss)
        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)
        # Append validation values to dictionary
        train_history['validation'].append(validation_metric) # [total, metricp, metrice]
        # Append times per epoch to dictionary
        train_history['time4epoch'].append(t_epoch)

    #     pickle.dump({'train': train_history, 'test': test_history}, open(params['results_dir'] + '/TrainHist/histdict_{}.pkl'.format(gan_no), 'wb'))
        tf.saved_model.save(gen.gen, os.path.join(save_dir, 'Generators', 'gen_{}'.format(step), 'gen_{}_ep_{}'.format(step, epoch)))
        tf.saved_model.save(disc.disc, os.path.join(save_dir, 'Discriminators', 'disc_{}'.format(step), 'disc_{}_ep_{}'.format(step, epoch)))

        loss_table(params, train_history, test_history, params['results_dir'], epoch, validation_metric[2], save=False, time4epoch = t_epoch) 

    # write into file the validation values (total val)
    f = open(os.path.join(save_dir, "validation_info_{}.txt".format(step)),"w")
    f.write('\nStart epoch: {} Validation: {}'.format(start_epoch, gan_bestval[step]))
    f.write('\nNew epochs validation:')
    for i in range(len(train_history['validation'])):
        f.write('\n{}'.format(train_history['validation'][i]))
    f.close()    