# AdaGAN class to wrap the ensemble training

from curses import KEY_MARK
from distutils.ccompiler import new_compiler
import numpy as np
# from tensorflow.python.framework.ops import name_scope
from datetime import datetime
import copy


# from sklearn.utils import shuffle

# from Models_m4_1_ReLU import generator_LeakyReLU
from gan_v6 import *
import data_handler_v6 as datah

# import tensorflow as tf

# gpus = tf.config.list_physical_devices('GPU')

class Adagan(object):
    """
    class to wrap the ensemble of generators and discriminator
    """

    def __init__(self, params):
        # initial setup
        self._steps_total = params['num_generators']
        self._steps_made = params['steps_made']
        self.data_obj = datah.EnergData(params)

        self._beta_vec = list()
        self._gens_bestepoch = list()
        self._mixdiscs_bestepoch = list()

        self.gens_list = []
        
        if self._steps_made > 0:
            filename = params['results_dir'] + '/Info/adagan_info.pkl'
            f = open(filename, 'rb')
            self.dict_adagan = pickle.load(f)
            f.close()

            # Load information about best epochs for GAN and mixdisc, times, bestval, etc.
            self._gens_bestepoch = copy.deepcopy(self.dict_adagan['gan_bestepoch'])
            self._mixdiscs_bestepoch = copy.deepcopy(self.dict_adagan['mixdisc_bestepoch'])

            # THIS IS UPDATED
            # Load beta vectors in the final setup
            if len(self.dict_adagan['beta_vec'][-1]) > self._steps_made:
                self._beta_vec = copy.deepcopy(self.dict_adagan['beta_vec'][-2])
                self.dict_adagan['beta_vec'] = self.dict_adagan['beta_vec'][:-1]
            else:
                self._beta_vec = copy.deepcopy(self.dict_adagan['beta_vec'][-1])
        else:
            self.dict_adagan = defaultdict(list)
            self.dict_adagan['params'].append(params)
            f = open(params['results_dir'] + "/loss_table.txt", "w")
            # f = open(os.path.join(params['results_dir'], "/loss_table.txt"),"w")
            now = datetime.now()
            f.write(now.strftime("%d/%m/%Y %H:%M:%S"))
            f.write("\n\n")
            f.close()
        return

    def prepare_datainfo(self, params):
        """ Get the number and shape of training data (load them if all can be loaded to memory)
        """
        self.data_obj.get_shape_num(params)
        return

    def train_ensemble(self, params): # !!! Unfinished - add some plots !!!
        """ Function to wrap training multiple generators
            1) Get mixture parameter beta - DONE
            2) Get predictions from mixture discriminator (probably in the gan class - just generate data and send them over there)
            3) Get lambda* (function of adagan)
            4) Get new weights of training data (function of adagan)
            5) Train new GAN (gen-disc pair, in the gan class)
            6) Save generator into gen_list
            7) Train the mixture discriminator
        """ 

        for step in range(self._steps_made, self._steps_total):
            beta = self._next_beta(params) # mixture weight beta_t; if step 0 - returns 1
            logging.info('Beta param for GAN {}: {:.2f}'.format(self._steps_made, beta))

            # Update data weights - if it is the first component or bagging, do nothing (weights are uniform)
            data_weights = None
            indices_train_list = None
            indices_test_list = None
            print('adaGAN step: {}'.format(self._steps_made))
            if self._steps_made > 0 and not params['is_bagging']:
                tick_mixdisc = time.time()
                # Train MIXDISC
                tick_mix_train = time.time()
                mixdisc_bestepoch, train_loss, test_loss = self.train_mix_discriminator(params)
                tock_mix_train = time.time()
                # Get True/Fake predictions from MIXDISC
                tick_mix_preds = time.time()
                mixdisc_ratios, indices_train_list, indices_test_list = self._get_predictions(params, self._steps_made, mixdisc_bestepoch)
                tock_mix_preds = time.time()
                # Compute weights for training data
                tick_mix_weights = time.time()
                data_weights = self._compute_data_weights_star(beta, mixdisc_ratios, indices_train_list)
                tock_mix_weights = time.time()
                tock_mixdisc = time.time()
                t_mixdisc = int(tock_mixdisc - tick_mixdisc)
                t_mix_train = int(tock_mix_train - tick_mix_train)
                t_mix_preds = int(tock_mix_preds - tick_mix_preds)
                t_mix_weights = int(tock_mix_weights - tick_mix_weights)
                logging.info('Time to process MIXDISC: {:02d}:{:02d}:{:02d}'.format(t_mixdisc // 3600, (t_mixdisc % 3600 // 60), t_mixdisc % 60))
                logging.info('Time to train MIXDISC: {:02d}:{:02d}:{:02d}'.format(t_mix_train // 3600, (t_mix_train % 3600 // 60), t_mix_train % 60))
                logging.info('Time to get predictions MIXDISC: {:02d}:{:02d}:{:02d}'.format(t_mix_preds // 3600, (t_mix_preds % 3600 // 60), t_mix_preds % 60))
                logging.info('Time to get weights: {:02d}:{:02d}:{:02d}'.format(t_mix_weights // 3600, (t_mix_weights % 3600 // 60), t_mix_weights % 60))
                # Save the data weights for further analysis
                pickle.dump(data_weights, open(params['results_dir'] + '/DataWeights/dataw_step_{}.pkl'.format(self._steps_made), 'wb'))
                pickle.dump(indices_train_list, open(params['results_dir'] + '/DataWeights/indices_step_{}.pkl'.format(self._steps_made), 'wb'))

            # Train GAN
            with Gan(params) as gan:
                tick_gan = time.time()
                logging.info('GAN {} training'.format(step))
                gan_bestepoch, gan_bestval = gan.train_gan(params, self.data_obj, self._steps_made, \
                                                data_weights, indices_train_list, indices_test_list) # returns index of the best training epoch
                tock_gan = time.time()
                t_gan = int(tock_gan - tick_gan)
                logging.info('Time to train GAN: {:02d}:{:02d}:{:02d}'.format(t_gan // 3600, (t_gan % 3600 // 60), t_gan % 60))
            
            # Rescale beta params
            if self._steps_made == 0:
                self._beta_vec = np.array([beta])
            else:
                rescaled_old = [v* (1.0 - beta) for v in self._beta_vec] # rescale the old beta weights to sum to 1 with the new beta
                self._beta_vec = np.array(rescaled_old + [beta])
            
            # Save weights, print info, plot loss graphs
            # params['results_dir']
            # Record the sequence of the best epochs for both GAN and MIXDISC
            self.dict_adagan['beta_vec'].append(self._beta_vec)
            self.dict_adagan['gan_bestepoch'].append(gan_bestepoch)
            self.dict_adagan['gan_bestval'].append(gan_bestval)
            self.dict_adagan['t_gan'].append(t_gan)
            if self._steps_made > 0 and not params['is_bagging']:
                self.dict_adagan['mixdisc_bestepoch'].append(mixdisc_bestepoch)
                self.dict_adagan['mixdisc_best_trainloss'].append(train_loss)
                self.dict_adagan['mixdisc_best_testloss'].append(test_loss)
                self.dict_adagan['mixdisc_total_time'].append(t_mixdisc)
                self.dict_adagan['mixdisc_train_time'].append(t_mix_train)
                self.dict_adagan['mixdisc_weights_time'].append(t_mix_weights)
                self.dict_adagan['mixdisc_preds_time'].append(t_mix_preds)
            pickle.dump(self.dict_adagan, open(params['results_dir'] + '/Info/adagan_info.pkl', 'wb'))
            self._gens_bestepoch.append(gan_bestepoch)

            self._steps_made += 1

        return


    def _next_beta(self, params): # DONE - adding more beta options is possible
        """
        Compute mixture weight according to params['beta_heur'] algorithm
        - constant = fixed number
        - uniform = all generators have the same weight
        - r constant = using r ration, r = const.
        - r decreasing = using exp. decreasing r ration
        """
        if self._steps_made == 0:
            beta = 1
        else:
            if params['beta_heur'] == 'constant':
                # Assign fixed number to beta
                assert params['beta_constant'] >=0, 'Beta should be nonnegative'
                assert params['beta_constant'] < 1, 'Beta should be < 1'
                beta = params['beta_constant']
            elif params['beta_heur'] == 'uniform' or params['is_bagging']:
                # All generators with the same weight alpha_t = 1/T; T = num_generators
                beta = 1./(self._steps_made + 1.)
            else:
                assert False, 'Unknown beta heuristic'
        return beta


    def _compute_data_weights_star(self, beta, ratios, indices_train):
        """Theory-inspired reweighting of training points.
        Refer to Section 3.1 of the arxiv paper

        ratios is a list of list (or np arrays?), each for one data file
        indices_train = list (each list element is a vector for a given file)
        """
        flattened = []
        for i in range(len(ratios)):
            flattened.extend(ratios[i])
        flattened = np.array(flattened)
        ratios_sorted = np.sort(flattened)
        cumsum_ratios = np.cumsum(ratios_sorted)
        is_found = False
        num = len(ratios_sorted)
        # We first find the optimal lambda* which is guaranteed to exits.
        # While Lemma 5 guarantees that lambda* <= 1, in practice this may
        # not be the case, as we replace dPmodel/dPdata by (1-D)/D.
        for i in range(num):
            # Computing lambda from equation (18) of the arxiv paper
            _lambda = beta * num * (1. + (1.-beta) / beta \
                    / num * cumsum_ratios[i]) / (i + 1.)
            if i == num - 1:
                if _lambda >= (1. - beta) * ratios_sorted[-1]:
                    is_found = True
                    break
            else:
                if _lambda <= (1 - beta) * ratios_sorted[i + 1] \
                        and _lambda >= (1 - beta) * ratios_sorted[i]:
                    is_found = True
                    break
        # Next we compute the actual weights using equation (17)
        data_weights = np.zeros(num)
        if is_found:
            _lambdamask = flattened <= (_lambda / (1.-beta))
            _lambdamask = np.squeeze(_lambdamask)
            flattened = np.squeeze(flattened)
            data_weights[_lambdamask] = (_lambda -
                                         (1-beta)*flattened[_lambdamask]) / num / beta
            logging.info(
                'Lambda={}, sum={}, deleted points={}'.format(
                    _lambda,
                    np.sum(data_weights),
                    1.0 * (num - sum(_lambdamask)) / num))
            # This is a delicate moment. Ratios are supposed to be
            # dPmodel/dPdata. However, we are using a heuristic
            # esplained around (16) in the arXiv paper. So the
            # resulting weights do not necessarily need to some
            # to one.
            data_weights = data_weights / np.sum(data_weights)
            # Split the weights according to the data files
            weights_list = []
            processed = 0
            for i in range(len(ratios)):
                num_end = processed + len(indices_train[i])
                weights_list.append(data_weights[processed:num_end].tolist())
                processed = num_end
            return weights_list
        else:
            logging.debug(
                '[WARNING] Lambda search failed, passing uniform weights')
            data_weights = np.ones(num) / (num + 0.)
            weights_list = []
            processed = 0
            for i in range(len(ratios)):
                num_end = processed + len(indices_train[i])
                weights_list.append(data_weights[processed:num_end].tolist())
                processed = num_end
            return weights_list


    def _get_predictions(self, params, mixdisc_no, epoch_no):
        """ Loop over the data files -> run prediction -> save into list of lists
        """
        ratios_list = []
        indices_train_list = []
        indices_test_list = []
        nb_files = len(self.data_obj.filenames)
        logging.info("Training predictions from MIXDISC")
        # Load the mixdisc architecture and weights
        with Discriminator(params) as mixdisc:
            weights_path = params['results_dir'] + '/Weights/mixdisc/mixdisc_{}/params_mixdisc_epoch_{}.hdf5'.format(mixdisc_no, epoch_no)
            mixdisc.disc.load_weights(weights_path)

            for file_no in range(nb_files):
                print("File: {}/{}".format(file_no+1, nb_files))
                self.data_obj.load_file(params, file_no)
                X_train, indices_train, indices_test = self.data_obj.prepare_images_w(params)
                # predictions in batches
                batch_size = params['batch_size']
                num_train = np.shape(X_train)[0]
                preds = []
                nb_batches = int(num_train / batch_size)
                # print('number of training data: {}'.format(np.shape(X_train)))
                for batch_no in range(nb_batches):
                    if batch_no == nb_batches-1:
                        image_batch  = X_train[(batch_no*batch_size) : ]
                    else:
                        image_batch  = X_train[(batch_no*batch_size) : ((batch_no+1)*batch_size)]
                    preds_batch,_,_ = mixdisc.disc(image_batch)
                    # preds_batch is fp16 or fp32
                    
                    preds.extend(preds_batch)
                # preds,_,_ = mixdisc.disc(X_train)
                preds = np.array(preds)
                # what happens in numpy

                # print(np.shape(preds))
                ratios = (1. - preds) / (preds + 1e-8)
                ratios_list.append(ratios.tolist())
                indices_train_list.append(indices_train)
                indices_test_list.append(indices_test)

        return ratios_list, indices_train_list, indices_test_list


    def train_mix_discriminator(self, params):
        """ Train discriminator on a mixture of images.
        """
        train_history = defaultdict(list)   #create a dict of lists
        test_history = defaultdict(list)

        total_train_loss = []
        total_test_loss = []

        logging.info('Start of MIXDISC training...')
        with Mixdisc(params, self._steps_made, self._gens_bestepoch) as mixdisc:

            for epoch in range(1,params['DGAN_epochs']+1):
                logging.info('Epoch {} of {}'.format(epoch, params['DGAN_epochs']))
                tick_epoch = time.time()

                # loss list
                epoch_loss = []

                if 'X_test' in locals():
                    del X_test
                    del Ep_test
                    del ecal_test
                
                nb_files = len(self.data_obj.filenames)
                for file_no in range(nb_files):
                    tick_import = time.time()
                    print("File: {}/{}".format(file_no+1, nb_files))
                    self.data_obj.load_file(params, file_no)
                    X_train, X_test_file, Ep_train, Ep_test_file, ecal_train, ecal_test_file = self.data_obj.data_preparation(params)
                    tock_import = time.time()
                    t_import = int(tock_import-tick_import)
                    print('Time for Import: {:02d}:{:02d}:{:02d}'.format(t_import // 3600, (t_import % 3600 // 60), t_import % 60))

                    if 'X_test' in locals():
                        X_test = np.concatenate((X_test, X_test_file))
                        Ep_test = np.concatenate((Ep_test, Ep_test_file))
                        ecal_test = np.concatenate((ecal_test, ecal_test_file))
                    else: 
                        X_test = X_test_file
                        Ep_test = Ep_test_file
                        ecal_test = ecal_test_file

                    # learning rate
                    lr_d = l_dec(params['lrate_d'], epoch*20+file_no-20) # it decays depending on the epoch and the number of the file
                    optimizer_d = tf.optimizers.Adam(lr_d)
                    print("Learnrate Discriminator: {:.4f}".format(lr_d))

                    nb_batches = int(X_train.shape[0] / params['batch_size'])
                    if params['verbose']:
                        progress_bar = tf.keras.utils.Progbar(target=nb_batches)

                    print("MIXDISC training - number of batches: {}".format(nb_batches))
                    # TRAINING IN BATCHES
                    for batch_no in range(nb_batches):
                        if params['verbose']:
                            progress_bar.update(batch_no+1)
                        else:
                            if epoch % 100 == 0:
                                print('processed {}/{} batches'.format(batch + 1, nb_batches))
                        
                        # Create train batches
                        batch_size = params['batch_size']
                        image_batch  = X_train[(batch_no*batch_size) : ((batch_no+1)*batch_size)]
                        energy_batch = Ep_train[(batch_no*batch_size) : ((batch_no+1)*batch_size)]
                        ecal_batch   = ecal_train[(batch_no*batch_size) : ((batch_no+1)*batch_size)]

                        generated_images, gen_aux, gen_ecal = self.generate(params, mixdisc.gens_list, batch_size, all_out = True)
                        d_loss = mixdisc.disc.train_step(params, optimizer_d, image_batch, energy_batch, ecal_batch, \
                                                        generated_images, gen_aux, gen_ecal)
                        epoch_loss.append(d_loss)

                # TESTING IN BATCHES
                nb_batches = int(X_test.shape[0] / batch_size)
                if nb_batches == 0:  #I need this or I get an error if I reduce the data with percentage
                    nb_batches = 1
                # Prepare list for saving losses
                disc_test_loss_list = []
                for batch in range(nb_batches):
                    #create batches
                    image_batch  = X_test[(batch*batch_size) : ((batch+1)*batch_size)]
                    energy_batch = Ep_test[(batch*batch_size) : ((batch+1)*batch_size)]
                    ecal_batch   = ecal_test[(batch*batch_size) : ((batch+1)*batch_size)]
                    # Get test loss
                    generated_images, gen_aux, gen_ecal = self.generate(params, mixdisc.gens_list, batch_size, all_out = True)
                    d_test_loss = mixdisc.disc.test_step(params, image_batch, energy_batch, ecal_batch, \
                                                    generated_images, gen_aux, gen_ecal)

                    disc_test_loss_list.append(d_test_loss)

                discriminator_train_loss = np.mean(np.array(epoch_loss), axis=0)   # average losses over batches (average losses of the given epoch)
                discriminator_test_loss = np.mean(np.array(disc_test_loss_list), axis=0)

                tock_epoch = time.time()
                t_epoch = int(tock_epoch-tick_epoch)
                logging.info('Mixdisc: Time for Epoch: {:02d}:{:02d}:{:02d}'.format(t_epoch // 3600, (t_epoch % 3600 // 60), t_epoch % 60))
                
                del X_train
                del X_test
                del X_test_file
                del Ep_train
                del Ep_test
                del Ep_test_file
                del ecal_train
                del ecal_test
                del ecal_test_file

                # Append epoch losses, timings to dictionary
                train_history['mixdisc'].append(discriminator_train_loss)
                test_history['mixdisc'].append(discriminator_test_loss)
                train_history['time4epoch'].append(t_epoch)
                
                # Pass total loss to list
                total_train_loss.append(discriminator_train_loss[0])
                total_test_loss.append(discriminator_test_loss[0])
                
                # Save the hist dictionary
                pickle.dump({'train': train_history, 'test': test_history}, open(params['results_dir'] + '/TrainHist/histdict_mixdisc_{}.pkl'.format(self._steps_made), 'wb'))

                # Save weights
                # self.gen.save_weights(params['results_dir']+"/Weights/gen/gen_{}/params_generator_epoch_{}.hdf5".format(gan_no, epoch),overwrite=True)
                mixdisc.disc.disc.save_weights(params['results_dir']+"/Weights/mixdisc/mixdisc_{}/params_mixdisc_epoch_{}.hdf5".format(self._steps_made, epoch),overwrite=True)

                if True:
                    f= open(params['results_dir'] + "/loss_table.txt","a")
                    f.write("\n")
                    f.write("MIXDISC epoch {}:".format(epoch))
                    e = t_epoch
                    f.write('\nTime for Epoch: {:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
                    f.write("\nTest loss: " + str(discriminator_test_loss[0]))
                    f.write('-' * 65)
                    f.write("\n\n")
                    f.close()
            # Here we have the end of a loop over epochs

        # Choose the best discriminator
        best_index_train = total_train_loss.index(np.min(total_train_loss))
        best_index = total_test_loss.index(np.min(total_test_loss))
        logging.info("Mixdisc - Best Weights Epoch: {}".format(best_index+1))
        self._mixdiscs_bestepoch.append(best_index+1)

        logging.info('MIXDISC trained. Best epoch: {}'.format(best_index+1))
        return best_index+1, total_train_loss[best_index_train], total_test_loss[best_index]


    def generate(self, params, generators_list, nb_samples=None, all_out=False, beta_replace=None, input_Ep=None):
        # Randomly select how many samples will be generated from each component (individual generator)
        # all_out ... if True, images, aux and ecal are returned. If False, only images are returned.
        # generators_list ... list of generators (weights already loaded)
        # nb_samples ... number of samples to generate
        # beta_replace ... If None, all available generators are used. If list of beta values, then the first len(beta) generators are used with the specified beta params.
        # input_Ep ... enables to specify the Ep primary energy of particles. If None, Ep is randomly selected from energy_range = (0.02, 5)

        if beta_replace is None:
            beta_vec = self._beta_vec
        else:
            beta_vec = beta_replace
        num_components = len(beta_vec) # number of generators
        component_ids = []
        if nb_samples is None:
            if input_Ep is None:    # check if nb_samples or input_Ep were specified
                print("No nb_samples or input_Ep was passed, skipping generate() function.")
                return None, None, None
            else:
                nb_samples = len(input_Ep)  # nb_samples determined from the Ep
            
        for _ in range(nb_samples):
            new_id = np.random.choice(np.arange(num_components), 1,
                                      p=beta_vec)[0]
            component_ids.append(new_id)
        points_per_component = [component_ids.count(i)
                                for i in range(num_components)] # how many samples to generate from each component

        # Next we sample required number of points per component
        sample_list = []
        aux_list = []
        ecal_list = []
        processed = 0 # number of generated samples - because of Ep
        for comp_id  in range(num_components): # component_id
            num = points_per_component[comp_id]
            if num == 0:
                continue
            # Generate data
            if input_Ep is None:
                comp_samples, gen_aux, gen_ecal = generators_list[comp_id].generate(num)
                aux_list.append(gen_aux)
                ecal_list.append(gen_ecal)
            else:
                noise = np.random.normal(0, 1, (len(input_Ep), params['latent_size']))
                gen_inp = np.multiply(input_Ep.reshape((-1, 1)), noise)
                comp_samples = generators_list[comp_id].generate(num, generator_input = gen_inp[processed:processed+num])
                processed = processed+num
            # comp_samples, gen_aux, gen_ecal = self._gens[comp_id].generate(_num)
            sample_list.append(np.array(comp_samples))
        if all_out:
            sample = sample_list[0]
            aux = aux_list[0]
            ecal = ecal_list[0]
            for i in range(1,len(aux_list)):
                sample = np.concatenate((sample, sample_list[i]), axis=0)
                aux = np.concatenate((aux, aux_list[i]), axis=0)
                ecal = np.concatenate((ecal, ecal_list[i]), axis=0)
            shuffle_index = np.random.permutation(sample.shape[0])
            return np.array(sample[shuffle_index]), np.array(aux[shuffle_index]), np.array(ecal[shuffle_index])
        else:
            sample = sample_list[0]
            for i in range(1,len(sample_list)):
                sample = np.concatenate((sample, sample_list[i]), axis=0)
            shuffle_index = np.random.permutation(sample.shape[0])
            return np.array(sample[shuffle_index]), np.array(input_Ep[shuffle_index]), None


    def generate_samples(self, params, num_samples = None, gan_nums=None, input_Ep = None):         # save_dir=None, save=False
        """
        Generate samples from the mixture of generators with mixture weights self._beta_vec
        gan_nums ... how many generators to use for the smaples creation
        """
        # for _ in range(num):
        #     new_id = np.random.choice(self._steps_made, 1, p=self._beta_vec)
        # Adjust beta_vec according to the number of generators we want to use
        if gan_nums is None: # use all available generators
            beta_vec = None
            gan_nums = self._steps_made
        else:
            filename = params['results_dir'] + '/Info/adagan_info.pkl'
            f = open(filename, 'rb')
            info = pickle.load(f)
            f.close()
            beta_vec = info['beta_vec'][gan_nums-1]
            print('Beta vector: {}'.format(beta_vec))
        # Load generator architectures and best weights
        if self.gens_list == []:
            for i in range(gan_nums):
                self.gens_list.append(Generator(params, arch_overwrite="FloG1"))
                self.gens_list[i].load_w_epoch(params, i, self._gens_bestepoch[i])
        if input_Ep is None:
            sample, aux, ecal = self.generate(params, self.gens_list, num_samples, all_out=True, beta_replace=beta_vec)
            return np.squeeze(sample), aux*100, ecal
        else:
            sample, ep_shuffeled, _ = self.generate(params, self.gens_list, num_samples, all_out=False, beta_replace=beta_vec, input_Ep=input_Ep/100)
            sample = np.squeeze(sample) # reshape to (num, xdim, ydim, zdim)
            ecal = np.sum(sample, axis=(1,2,3))
            return sample, ep_shuffeled*100, ecal   # ecal can be calculated afterwards


    def load_generators(self, params, gan_nums=None, weights_only=True, path=None):
        """
        Function to load generators into the self.gens_list
        weights_only ... if True, it gets the architecture from the gan_v6.py file and loads weights of the trained model
                            if False, loads models from .pb file
        """
        if gan_nums is None:
            gan_nums = self._steps_made
        if weights_only == True:
            for i in range(gan_nums):
                self.gens_list.append(Generator(params, arch_overwrite="FloG1"))
                self.gens_list[i].load_w_epoch(params, i, self._gens_bestepoch[i])
        else:
            if path is None:
                for i in range(gan_nums):
                    self.gens_list.append(Generator(params, arch_overwrite="FloG1"))
                    self.gens_list[i].load_pb_epoch(params, i, self._gens_bestepoch[i])
            else:
                path = path if isinstance(path, list) else [path]
                for i in range(len(path)):
                    self.gens_list.append(Generator(params, arch_overwrite="FloG1"))
                    self.gens_list[i].load_pb_epoch(params, path=path[i])
        return

