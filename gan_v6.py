###################################################
#  Gan architectures and custom loss functions
#  + learning rate decay function
###################################################

#file which contains neural network models and loss functions - same as Models_m4_1_ReLU.py

#v4_1: includes a second network with ReLU function as last layer

import tensorflow as tf
import numpy as np
import time
import sys
import math as math
import logging
import pickle
from collections import defaultdict
import gc
from func_utils_v6 import *

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')

if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_memory_growth(gpus[1], True)
    tf.config.set_visible_devices(gpus[1], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

# try:
#   tf.config.experimental.set_memory_growth(gpus[1], True)
# except:
#   # Invalid device or cannot modify virtual devices once initialized.
#   pass

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     for gpu in gpus:
#         tf.config.experimental.set_virtual_device_configuration(gpu,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=24480)])

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

############################################################################

def l_dec(initial_lrate, epoch, start_decay=80, decay_rate=0.012):
    """ Decaying learning rate
    """
    epoch = epoch - 1 #because training starts at epoch 1
    if epoch < start_decay:
        k = 0.0
    else:
        k = decay_rate #0.07
        epoch = epoch - start_decay
    lrate = initial_lrate * math.exp(-k*epoch)
    return lrate

def bit_flip_tf(x, prob = 0.05):
    """ Flips a int array's values with some probability """
    x = np.array(x)
    selection = np.random.uniform(0, 1, x.shape) < prob
    x[selection] = 1* np.logical_not(x[selection])
    x = tf.constant(x)
    return x

##########################################################################


class Discriminator(object):
    """ Discriminator architecture, training procedure, testing, running predictions?
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Cleaning the whole default Graph
        logging.debug('Cleaning the graph...')
        logging.debug('Closing the session...')

    def __init__(self, params):
        self._disc_arch = params['disc_arch']
        self.disc = self._discriminator(params)
        logging.debug('Discriminator initialized, architecture {} loaded.'.format(self._disc_arch))
        # self.disc.summary()
        return

# d_loss = self.disc.train_step(params, lr_d, image_batch, energy_batch, ecal_batch, \
#                                                     generated_images, gen_aux, gen_ecal)

    def load_w_epoch(self, params, disc_no, e):
        # dweights = params['results_dir'] + "/Weights/disc/disc_{}/params_discriminator_epoch_{}.hdf5".format(disc_no,e)
        dweights = params['results_dir'] + "/Weights/disc/disc_{}/params_generator_epoch_{}.hdf5".format(disc_no,e)   # mistake in the file naming - corrected but stays in the old results
        # print(dweights)
        self.disc.load_weights(dweights)
        return
    
    def train_step(self, params, optimizer_d, image_batch, energy_batch, ecal_batch, image_gen, energy_gen, ecal_gen):
        # optimizer_d = tf.optimizers.Adam(lrate)

        #discriminator true training
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.disc.trainable_variables)
            d_loss_true = self._disc_loss(image_batch, energy_batch, ecal_batch, label ="ones", \
                                            wtf=params['wtf'], wa=params['wa'], we=params['we'])
            d_grads = tape.gradient( d_loss_true[0] , self.disc.trainable_variables )
        optimizer_d.apply_gradients( zip( d_grads , self.disc.trainable_variables) )

        #discriminator fake training
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.disc.trainable_variables)
            d_loss_fake = self._disc_loss(image_gen, energy_gen, ecal_gen, label = "zeros", \
                                            wtf=params['wtf'], wa=params['wa'], we=params['we'])
            d_grads = tape.gradient( d_loss_fake[0] , self.disc.trainable_variables )
        optimizer_d.apply_gradients( zip( d_grads , self.disc.trainable_variables) )

        d_loss = []
        d_loss.append( (d_loss_true[0] + d_loss_fake[0])/2)
        d_loss.append( (d_loss_true[1] + d_loss_fake[1])/2)
        d_loss.append( (d_loss_true[2] + d_loss_fake[2])/2)
        d_loss.append( (d_loss_true[3] + d_loss_fake[3])/2)

        return [d_loss[0].numpy(), d_loss[1].numpy(), d_loss[2].numpy(), d_loss[3].numpy()]

    def test_step(self, params, image_batch, energy_batch, ecal_batch, image_gen, energy_gen, ecal_gen):
        d_test_loss_true = self._disc_loss(image_batch, energy_batch, ecal_batch, label ="ones", \
                                            wtf=params['wtf'], wa=params['wa'], we=params['we'])
        d_test_loss_fake = self._disc_loss(image_gen, energy_gen, ecal_gen, label = "zeros", \
                                            wtf=params['wtf'], wa=params['wa'], we=params['we'])
        d_test_loss = []
        d_test_loss.append( (d_test_loss_true[0] + d_test_loss_fake[0])/2)
        d_test_loss.append( (d_test_loss_true[1] + d_test_loss_fake[1])/2)
        d_test_loss.append( (d_test_loss_true[2] + d_test_loss_fake[2])/2)
        d_test_loss.append( (d_test_loss_true[3] + d_test_loss_fake[3])/2)
        return [d_test_loss[0].numpy(), d_test_loss[1].numpy(), d_test_loss[2].numpy(), d_test_loss[3].numpy()]
    

    def _disc_loss(self, image_batch, energy_batch, ecal_batch, label, wtf=6.0, wa=0.2, we=0.1, mixd = False):
        """ Discriminator loss: 4 components (Total, True/Fake, aux (Ep regression task), ECAL)
        """

        discriminate = self.disc(image_batch)

        #true/fake loss
        if label == "ones":
            labels = bit_flip_tf(tf.ones_like(discriminate[0])*0.9)     #true=1    
        elif label == "zeros":
            labels = bit_flip_tf(tf.zeros_like(discriminate[0])*0.1)    #fake=0
        loss_true_fake = tf.reduce_mean(- labels * tf.math.log(discriminate[0] + 2e-7) - (1 - labels) * tf.math.log(1 - discriminate[0] + 2e-7))

        #aux loss
        loss_aux = tf.reduce_mean(tf.math.abs((energy_batch - discriminate[1])/(energy_batch + 2e-7))) *100
            
        #ecal loss
        loss_ecal = tf.reduce_mean(tf.math.abs((ecal_batch - discriminate[2])/(ecal_batch + 2e-7))) *100
    
        #total loss
        weight_true_fake = wtf
        weight_aux = wa
        weight_ecal = we
        total_loss = weight_true_fake * loss_true_fake + weight_aux * loss_aux + weight_ecal * loss_ecal
        return total_loss, loss_true_fake, loss_aux, loss_ecal


    def _discriminator(self, params):
        #keras_dformat='channels_first'
        if params['keras_dformat'] =='channels_last':
            dshape=(25, 25, 25,1)
            daxis=(1,2,3)
            axis = -1 
        else:
            dshape=(1, 25, 25, 25)
            daxis=(2,3,4)
            axis = 1 
        #keras_dformat='channels_first'   #i need this when I train gen with ch last and keras with ch first
        #dshape=(25, 25, 25, 1)    
        image = tf.keras.layers.Input(shape=dshape, dtype="float32")     #Input Image
        x = image
        x = tf.keras.layers.Reshape([25,25,25])(x)
        
        # 3 rotations of the cube
        x1 = x
        x2 = tf.keras.layers.Permute([3,1,2])(x)   #permute starts indexing with 1
        x3 = tf.keras.layers.Permute([2,3,1])(x)   #permute starts indexing with 1
        
        def path(x):
            #path1
            #1.Conv Block
            x = tf.keras.layers.Conv2D(64, (8, 8), data_format=params['keras_dformat'], use_bias=False, padding='same')(x)
            x = tf.keras.layers.LeakyReLU() (x)
            x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
            x = tf.keras.layers.Dropout(0.2)(x)
            #2.Conv Block
            x = tf.keras.layers.Conv2D(32, (6, 6), data_format=params['keras_dformat'], padding='valid', use_bias=False, \
                                            kernel_initializer='he_uniform')(x)
            x = tf.keras.layers.LeakyReLU() (x)
            x = tf.keras.layers.BatchNormalization(axis=axis) (x)
            x = tf.keras.layers.Dropout(0.2)(x)
            #x = tf.keras.layers.MaxPooling2D((2, 2),strides=(2,2), data_format=keras_dformat)(x)
            #3.Conv Block
            x = tf.keras.layers.Conv2D(32, (5, 5), data_format=params['keras_dformat'], padding='valid', use_bias=False, 
                                            kernel_initializer='he_uniform')(x)
            x = tf.keras.layers.LeakyReLU() (x)
            x = tf.keras.layers.BatchNormalization(axis=axis) (x)
            x = tf.keras.layers.Dropout(0.2)(x)
            #4. Conv Block
            x = tf.keras.layers.Conv2D(32, (4, 4), data_format=params['keras_dformat'], padding='valid', use_bias=False, \
                                            kernel_initializer='he_uniform')(x)
            x = tf.keras.layers.LeakyReLU() (x)
            x = tf.keras.layers.BatchNormalization(axis=axis) (x)
            x = tf.keras.layers.Dropout(0.2)(x)
            #x = tf.keras.layers.MaxPooling2D((2, 2),strides=(2,2), data_format=keras_dformat)(x)
            #6. Conv Block
            x = tf.keras.layers.Conv2D(32, (3, 3), data_format=params['keras_dformat'], padding='valid', use_bias=False, 
                                            kernel_initializer='he_uniform')(x)
            x = tf.keras.layers.LeakyReLU() (x)
            x = tf.keras.layers.BatchNormalization(axis=axis) (x)
            x = tf.keras.layers.Dropout(0.2)(x)
            #7. Conv Block
            x = tf.keras.layers.Conv2D(9, (3, 3), data_format=params['keras_dformat'], padding='valid', use_bias=False, \
                                            kernel_initializer='he_uniform')(x)
            x = tf.keras.layers.LeakyReLU() (x)
            x = tf.keras.layers.BatchNormalization(axis=axis) (x)
            x = tf.keras.layers.Dropout(0.2)(x)
            #print(x.shape)
            return x

        x1 = path(x1)
        x2 = path(x2)
        x3 = path(x3)
        # print(x1.shape)
        
        # Rotate the cubes back and concatenate
        x2 = tf.keras.layers.Permute([2,3,1])(x2)   #permute starts indexing with 1
        x3 = tf.keras.layers.Permute([3,1,2])(x3)   #permute starts indexing with 1
        x = tf.keras.layers.concatenate([x1,x2,x3],axis=axis) #i stack them on the channels axis

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(10, activation='linear')(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.BatchNormalization(axis=axis)(x)

        #Takes Network outputs as input
        fake = tf.keras.layers.Dense(1, activation='sigmoid', name='generation')(x)   # True/Fake classifier
        aux = tf.keras.layers.Dense(1, activation='linear', name='auxiliary')(x)      # Should estimate the Ep
        # Takes image as input
        ecal = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, axis=daxis))(image)    # Sum of energies captured by the detector
        return tf.keras.Model(inputs=image, outputs=[fake, aux, ecal])





###################################################################



class Generator(object):
    """ Generator architecture, training procedure?, running predictions
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Cleaning the whole default Graph
        logging.debug('Cleaning the graph...')
        logging.debug('Closing the session...')

    def __init__(self, params, arch_overwrite = None):
        """ arch_overwrite gives the possibility to choose different architecture without changing the params list
        """
        if arch_overwrite is None:
            self._gen_arch = params['gen_arch']
        else:
            self._gen_arch = arch_overwrite

        if self._gen_arch == 'FloG1_leaky':
            self.gen = self._generator_LeakyReLU(params)
        elif self._gen_arch == 'FloG1':
            self.gen = self._generator_ReLU(params)
        else:
            assert False, 'Unknown generator architecture'
        logging.debug('Generator initialized, architecture {} loaded.'.format(self._gen_arch))
        # self.gen.summary()
        return

    def get_arch_gen(self, params, overwrite = None):
        """ Returns only the required generator architecture
        """
        if overwrite is None:
            which_arch = self._gen_arch
        else:
            which_arch = overwrite
        
        if which_arch == 'FloG1_leaky':
            return self._generator_LeakyReLU(params)
        elif which_arch == 'FloG1':
            return self._generator_ReLU(params)
        else:
            print("Unknown generator architecture.")
            return []


    def load_w_epoch(self, params, gan_no, e):
        gweights = params['results_dir'] + "/Weights/gen/gen_{}/params_generator_epoch_{}.hdf5".format(gan_no,e)
        # print(gweights)
        self.gen.load_weights(gweights)
        return

    def load_pb_epoch(self, params, gan_no=None, e=None, path=None):
        if path is None:
            if gan_no is None or e is None:
                raise ValueError('gan_no and epoch are not specified.')
            else:
                self.gen = tf.keras.models.load_model(os.path.join(params['results_dir'], 'Generators', 'gen_{}_ep_{}'.format(gan_no, e)))
        else:
            self.gen = tf.keras.models.load_model(path)
            # self.gen = tf.keras.models.load_model(os.path.join(path, 'gen_{}'.format(gan_no), 'gen_{}_ep_{}'.format(gan_no, e)))
        return


    def generate(self, num=None, epoch=10, batch_size=128, generator_input = None, latent=200):
        """ Generate samples
        """

        if generator_input is None and num is None:
            print('No num or generator_input passed to Generator.generate(). Skipping this function.')
            return None, None, None
        if generator_input is None:
            nb_batches = int(num / batch_size)
            for batch_no in range(nb_batches):
                # print('Generating batch_no: {}'.format(batch_no))
                # _, gen_aux, generator_input, gen_ecal = self._func_for_gen(nb_samples=num, epoch=epoch, latent_size=latent)
                _, gen_aux_batch, generator_input_batch, gen_ecal_batch = self._func_for_gen(nb_samples=batch_size, epoch=epoch, latent_size=latent) # first output is noise - not needed
                if batch_no == 0:
                    generated_images = self.gen(generator_input_batch)
                    gen_aux = np.copy(gen_aux_batch)
                    gen_ecal = np.copy(gen_ecal_batch)
                else:
                    generated_batch = self.gen(generator_input_batch)
                    generated_images = tf.concat([generated_images, generated_batch], 0)
                    gen_aux = np.concatenate((gen_aux, gen_aux_batch), axis=0)
                    gen_ecal = np.concatenate((gen_ecal, gen_ecal_batch), axis=0)
            # Generate images for the residual number
            num_residual = num % batch_size
            if not(num_residual == 0) and not(nb_batches == 0):
                _, gen_aux_batch, generator_input_batch, gen_ecal_batch = self._func_for_gen(nb_samples=num_residual, epoch=epoch, latent_size=latent) # first output is noise - not needed
                generated_batch = self.gen(generator_input_batch)            
                generated_images = tf.concat([generated_images, generated_batch], 0)
                gen_aux = np.concatenate((gen_aux, gen_aux_batch), axis=0)
                gen_ecal = np.concatenate((gen_ecal, gen_ecal_batch), axis=0)
            elif not(num_residual == 0) and nb_batches == 0:
                _, gen_aux_batch, generator_input_batch, gen_ecal_batch = self._func_for_gen(nb_samples=num_residual, epoch=epoch, latent_size=latent)
                generated_images = self.gen(generator_input_batch)
                gen_aux = np.copy(gen_aux_batch)
                gen_ecal = np.copy(gen_ecal_batch)
            return generated_images, gen_aux, gen_ecal
        else:
            nb_batches = int(generator_input.shape[0] / batch_size)
            for batch_no in range(nb_batches):
                input_batch  = generator_input[(batch_no*batch_size) : ((batch_no+1)*batch_size)]
                if batch_no == 0:
                    generated_images = self.gen(input_batch)
                else:
                    generated_batch = self.gen(input_batch)
                    generated_images = tf.concat([generated_images, generated_batch], 0)
            num_residual = generator_input.shape[0] % batch_size
            if not(num_residual == 0) and not(nb_batches == 0):
                input_batch = generator_input[(nb_batches*batch_size) : ]
                generated_batch = self.gen(input_batch)
                generated_images = tf.concat([generated_images, generated_batch], 0)
            elif not(num_residual == 0) and nb_batches == 0:
                generated_images = self.gen(generator_input)
            return generated_images


    def train_step(self, params, optimizer_g, batch_size, epoch, disc):
        # optimizer_g = tf.optimizers.Adam(lrate)

        #generator training
        gen_losses = []
        for i in range(1):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(self.gen.trainable_variables)
                g_loss = self._gen_loss(disc, batch_size=batch_size, epoch=epoch, \
                                            wtf=params['wtf'], wa=params['wa'], we=params['we'])
                g_grads = tape.gradient( g_loss[0] , self.gen.trainable_variables )
            optimizer_g.apply_gradients( zip( g_grads , self.gen.trainable_variables ) )
            # optimizer_g.apply_gradients( g_grads , self.gen.trainable_variables )
            gen_losses.append([g_loss[0].numpy(), g_loss[1].numpy(), g_loss[2].numpy(), g_loss[3].numpy()])

        return np.mean(gen_losses, axis = 0)


    def test_step(self, params, batch_size, epoch, disc):
        """ Get the generator loss on a test dataset
        """
        gen_test_loss = self._gen_loss(disc, batch_size, epoch=epoch, wtf=params['wtf'], wa=params['wa'], we=params['we'])
        return [gen_test_loss[0].numpy(), gen_test_loss[1].numpy(), gen_test_loss[2].numpy(), gen_test_loss[3].numpy()]


    def _gen_loss(self, disc, batch_size=128, epoch=10, wtf=6.0, wa=0.2, we=0.1):
        """ Generator loss: 3 components
        """
        generated_images, gen_aux, gen_ecal = self.generate(num = batch_size, epoch = epoch)

        discriminator_fake = disc.disc(generated_images)
        #true/fake
        label_fake = bit_flip_tf(tf.ones_like(discriminator_fake[0])*0.9)   #ones = true
        loss_true_fake = tf.reduce_mean(- label_fake * tf.math.log(discriminator_fake[0] + 2e-7) - 
                                (1 - label_fake) * tf.math.log(1 - discriminator_fake[0] + 2e-7))
        #aux
        loss_aux = tf.reduce_mean(tf.math.abs((gen_aux - discriminator_fake[1])/(gen_aux + 2e-7))) *100
        #ecal
        loss_ecal = tf.reduce_mean(tf.math.abs((gen_ecal - discriminator_fake[2])/(gen_ecal + 2e-7))) *100
        
        #total loss
        weight_true_fake = wtf
        weight_aux = wa
        weight_ecal = we
        total_loss = weight_true_fake * loss_true_fake + weight_aux * loss_aux + weight_ecal * loss_ecal
        return total_loss, loss_true_fake, loss_aux, loss_ecal


    def _func_for_gen(self, nb_samples, latent_size=200, epoch=10):
        """ Prepare input for the generator: latent vector, Ep, latent vector scaled by Ep
        """
        noise = np.random.normal(0, 1, (nb_samples, latent_size))  #input for bit_flip() to generate true/false values for discriminator
        if epoch<3:
            gen_aux = np.random.uniform(1, 4,(nb_samples,1 ))   # generates aux for dicriminator - basicaly the Ep/100 (i.e. Ep 100-400 GeV)
        else:
            gen_aux = np.random.uniform(0.02, 5,(nb_samples,1 ))   #generates aux for dicriminator - basicaly Ep/100 (i.e. Ep 2-500 GeV)
        #gen_ecal =         np.multiply(2, gen_aux)                          #generates ecal for discriminator
        generator_input =  np.multiply(gen_aux, noise)                      #generates input for generator - scale noise by "Ep"
        gen_ecal_func = self._GetEcalFit(gen_aux, mod=1)
        return noise, gen_aux, generator_input, gen_ecal_func


    # return a fit for Ecalsum/Ep for Ep 
    #https://github.com/svalleco/3Dgan/blob/0c4fb6f7d47aeb54aae369938ac2213a2cc54dc0/keras/EcalEnergyTrain.py#L288
    def _GetEcalFit(self, sampled_energies, particle='Ele', mod=0, xscale=1):
        # for this model set mod = 1 !!!!!
        #the smaller the energy, the closer is the factor to 2, the bigger the energy, the smaller is the factor
        """ ?????    (Function to estimate ECAL based on the Ep and particle type)
        """
        if mod==0:
            return np.multiply(2, sampled_energies)
        elif mod==1:
            if particle == 'Ele':
                root_fit = [0.0018, -0.023, 0.11, -0.28, 2.21]
                ratio = np.polyval(root_fit, sampled_energies)
                return np.multiply(ratio, sampled_energies) * xscale
            elif particle == 'Pi0':
                root_fit = [0.0085, -0.094, 2.051]
                ratio = np.polyval(root_fit, sampled_energies)
                return np.multiply(ratio, sampled_energies) * xscale


    def _generator_LeakyReLU(self, params):
        """ Architecture definition
        """
        #keras_dformat='channels_first'
        if params['keras_dformat'] == 'channels_first':
            axis = 1  #if channels first, -1 if channels last
            dshape = [1, 25, 25, 25]
        else:
            axis = -1
            dshape = [25, 25, 25, 1]
        dim = (5,5,5)

        latent = tf.keras.Input(shape=(params['latent_size']), dtype="float32")  # define Input     
        x = tf.keras.layers.Dense(5*5*5, input_dim=params['latent_size'])(latent)   #shape (none, 625) #none is batch size
        x = tf.keras.layers.Reshape(dim) (x)  # shape after (none, 5,5,5)  
        
        # 3 rotations
        x1 = x
        x2 = tf.keras.layers.Permute([3,1,2])(x)   #permute starts indexing with 1
        x3 = tf.keras.layers.Permute([2,3,1])(x)   #permute starts indexing with 1
        
        def path(x):
            #path1
            #1.Conv Block
            x = tf.keras.layers.Conv2D(32, (5, 5), data_format=params['keras_dformat'], use_bias=False, padding='same')(x)
            x = tf.keras.layers.LeakyReLU() (x)
            x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
            x = tf.keras.layers.Dropout(0.2)(x)
            #2.Conv Block
            x = tf.keras.layers.Conv2DTranspose(64, (5,5), strides =(3,3), data_format=params['keras_dformat'], padding="same") (x)
            x = tf.keras.layers.Conv2D(64, (5, 5), data_format=params['keras_dformat'], padding='same', use_bias=False, \
                                            kernel_initializer='he_uniform')(x)
            x = tf.keras.layers.LeakyReLU() (x)
            x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
            x = tf.keras.layers.Dropout(0.2)(x)
            #3.Conv Block
            x = tf.keras.layers.Conv2DTranspose(64, (5,5), strides =(2,2), data_format=params['keras_dformat'], padding="same") (x)
            x = tf.keras.layers.Conv2D(64, (8, 8), data_format=params['keras_dformat'], padding='same', use_bias=False, \
                                            kernel_initializer='he_uniform')(x)
            x = tf.keras.layers.LeakyReLU() (x)
            x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
            x = tf.keras.layers.Dropout(0.2)(x)
            #4. Conv Block
            x = tf.keras.layers.Conv2D(64, (5, 5), data_format=params['keras_dformat'], padding='valid', use_bias=False, \
                                            kernel_initializer='he_uniform')(x)
            x = tf.keras.layers.LeakyReLU() (x)
            x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
            x = tf.keras.layers.Dropout(0.2)(x)
            #5. Conv Block
            x = tf.keras.layers.Conv2D(32, (4, 4), data_format=params['keras_dformat'], padding='same', use_bias=False, \
                                            kernel_initializer='he_uniform')(x)
            x = tf.keras.layers.LeakyReLU() (x)
            x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
            x = tf.keras.layers.Dropout(0.2)(x)
            #6. Conv Block
            x = tf.keras.layers.Conv2D(32, (3, 3), data_format=params['keras_dformat'], padding='same', use_bias=False, \
                                            kernel_initializer='he_uniform')(x)
            x = tf.keras.layers.LeakyReLU() (x)
            x = tf.keras.layers.BatchNormalization(axis=axis) (x)  
            x = tf.keras.layers.Dropout(0.2)(x)
            #7.Conv Block
            x = tf.keras.layers.Conv2D(25, (2, 2), data_format=params['keras_dformat'], padding='valid', use_bias=False, \
                                            kernel_initializer='he_uniform')(x)
            x = tf.keras.layers.LeakyReLU() (x)
            x = tf.keras.layers.BatchNormalization(axis=axis) (x)  
            x = tf.keras.layers.Dropout(0.2)(x)
            return x

        x1 = path(x1)
        x2 = path(x2)
        x3 = path(x3)

        # Rotate the cubes back and concatenate
        x2 = tf.keras.layers.Permute([2,3,1])(x2)   #permute starts indexing with 1
        x3 = tf.keras.layers.Permute([3,1,2])(x3)   #permute starts indexing with 1
        x = tf.keras.layers.concatenate([x1,x2,x3],axis=axis) #i stack them on the channels axis
        
    #    print(x.shape)   
        x = tf.keras.layers.Conv2D(25, (3,3), data_format=params['keras_dformat'], padding='same', use_bias=False, \
                                        kernel_initializer='he_uniform')(x)
        
        x = tf.keras.layers.Reshape(dshape)(x)
        # x = tf.keras.layers.LeakyReLU() (x)
        return tf.keras.Model(inputs=[latent], outputs=x)   # Returns model with input and output specified


    def _generator_ReLU(self, params):
        
        #keras_dformat='channels_first'
        if params['keras_dformat'] == 'channels_first':
            axis = 1  #if channels first, -1 if channels last
            dshape = [1, 25, 25, 25]
        else:
            axis = -1
            dshape = [25, 25, 25, 1]
        dim = (5,5,5)

        latent = tf.keras.Input(shape=(params['latent_size']), dtype="float32")  #define Input     
        x = tf.keras.layers.Dense(5*5*5, input_dim=params['latent_size'])(latent)   #shape (none, 625) #none is batch size
        x = tf.keras.layers.Reshape(dim) (x)  #shape after (none, 5,5,5)  
        
        x1 = x
        x2 = tf.keras.layers.Permute([3,1,2])(x)   #permute starts indexing with 1
        x3 = tf.keras.layers.Permute([2,3,1])(x)   #permute starts indexing with 1
        
        def path(x):
            #path1
            #1.Conv Block
            x = tf.keras.layers.Conv2D(32, (5, 5), data_format=params['keras_dformat'], use_bias=False, padding='same')(x)
            x = tf.keras.layers.LeakyReLU() (x)
            x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
            x = tf.keras.layers.Dropout(0.2)(x)
            #2.Conv Block
            x = tf.keras.layers.Conv2DTranspose(64, (5,5), strides =(3,3), data_format=params['keras_dformat'], padding="same") (x)
            x = tf.keras.layers.Conv2D(64, (5, 5), data_format=params['keras_dformat'], padding='same', use_bias=False, \
                                            kernel_initializer='he_uniform')(x)
            x = tf.keras.layers.LeakyReLU() (x)
            x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
            x = tf.keras.layers.Dropout(0.2)(x)
            #3.Conv Block
            x = tf.keras.layers.Conv2DTranspose(64, (5,5), strides =(2,2), data_format=params['keras_dformat'], padding="same") (x)
            x = tf.keras.layers.Conv2D(64, (8, 8), data_format=params['keras_dformat'], padding='same', use_bias=False, \
                                            kernel_initializer='he_uniform')(x)
            x = tf.keras.layers.LeakyReLU() (x)
            x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
            x = tf.keras.layers.Dropout(0.2)(x)
            #4. Conv Block
            x = tf.keras.layers.Conv2D(64, (5, 5), data_format=params['keras_dformat'], padding='valid', use_bias=False, \
                                            kernel_initializer='he_uniform')(x)
            x = tf.keras.layers.LeakyReLU() (x)
            x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
            x = tf.keras.layers.Dropout(0.2)(x)
            #5. Conv Block
            x = tf.keras.layers.Conv2D(32, (4, 4), data_format=params['keras_dformat'], padding='same', use_bias=False, \
                                            kernel_initializer='he_uniform')(x)
            x = tf.keras.layers.LeakyReLU() (x)
            x = tf.keras.layers.BatchNormalization(axis=axis) (x) 
            x = tf.keras.layers.Dropout(0.2)(x)
            #6. Conv Block
            x = tf.keras.layers.Conv2D(32, (3, 3), data_format=params['keras_dformat'], padding='same', use_bias=False, \
                                            kernel_initializer='he_uniform')(x)
            x = tf.keras.layers.LeakyReLU() (x)
            x = tf.keras.layers.BatchNormalization(axis=axis) (x)  
            x = tf.keras.layers.Dropout(0.2)(x)
            #7.Conv Block
            x = tf.keras.layers.Conv2D(25, (2, 2), data_format=params['keras_dformat'], padding='valid', use_bias=False, \
                                            kernel_initializer='he_uniform')(x)
            x = tf.keras.layers.LeakyReLU() (x)
            x = tf.keras.layers.BatchNormalization(axis=axis) (x)  
            x = tf.keras.layers.Dropout(0.2)(x)
            return x

        x1 = path(x1)
        x2 = path(x2)
        x3 = path(x3)

        # Rotate the cubes back and concatenate
        x2 = tf.keras.layers.Permute([2,3,1])(x2)   #permute starts indexing with 1
        x3 = tf.keras.layers.Permute([3,1,2])(x3)   #permute starts indexing with 1
        x = tf.keras.layers.concatenate([x1,x2,x3],axis=axis) #i stack them on the channels axis
        
    #    print(x.shape)   
        x = tf.keras.layers.Conv2D(25, (3,3), data_format=params['keras_dformat'], padding='same', use_bias=False,  kernel_initializer='he_uniform')(x)
        
        x = tf.keras.layers.Reshape(dshape)(x)
        x = tf.keras.layers.ReLU() (x)
        return tf.keras.Model(inputs=[latent], outputs=x)   # model with input and output specified
        #generator().summary()
    


##################################################################


class Gan(object):
    """ Class to wrap GAN network architectures, training, loss functions, etc.
    """

    def __enter__(self):
        # tf.keras.backend.clear_session()
        # tf.keras.backend.clear_session()
        # gc.collect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Cleaning the whole default Graph
        # tf.keras.backend.clear_session()
        # gc.collect()
        logging.debug('Cleaning the graph...')
        logging.debug('Closing the session...')

    def __init__(self, params):
        """ Copy important parameters, initial setup
        """
        # mem_dict = tf.config.experimental.get_memory_info('GPU:0')
        # print('Memory usage at the beginning of gan init: {}'.format(mem_dict['current']))
        tf.keras.backend.clear_session()
        gc.collect()
        self.gen = Generator(params)
        # mem_dict = tf.config.experimental.get_memory_info('GPU:0')
        # print('Memory usage after gen is initialized: {}'.format(mem_dict['current']))
        self.disc = Discriminator(params)
        # mem_dict = tf.config.experimental.get_memory_info('GPU:0')
        # print('Memory usage after dict is also initialized: {}'.format(mem_dict['current']))
        # self.mixdisc = None
        logging.debug('GAN model initialized.')
        return


    def train_gan(self, params, data_obj, gan_no, data_weights = None, indices_train_list = None, indices_test_list = None):
        """ Train one GAN algorithm (generator-discriminator pair)
                Does not return anything
        """
        # Clean the gan and disc variables - to start training from the scratch
        # epoch is important for the decaying learning rate - function that we need to use
        
        # mem_dict = tf.config.experimental.get_memory_info('GPU:0')
        # print('Memory usage at the beginning of training: {}'.format(mem_dict['current']))

        train_history = defaultdict(list)   #create a dict with an empty list 
        test_history = defaultdict(list)
        for epoch in range(1, params['nb_epochs']+1): # start from epoch 1
            print('Epoch {} of {}'.format(epoch, params['nb_epochs']))
            tick_epoch = time.time()

            # loss lists
            epoch_gen_loss = []
            epoch_disc_loss = []

            if 'X_test' in locals():
                del X_test
                del Ep_test
                del ecal_test

            # Switch from LeakyReLU to ReLU generator
            if epoch == params['ReLU_epoch']:
                self.gen = Generator(params, arch_overwrite='FloG1')
                self.gen.load_w_epoch(params = params, gan_no = gan_no, e = str(params['ReLU_epoch']-1))

            nb_files = len(data_obj.filenames)
            print('nb_files : {}'.format(nb_files))
            for file_no in range(nb_files):
                tick_import = time.time()
                print("File: ", file_no+1, "/", nb_files)
                data_obj.load_file(params, file_no)
                if data_weights is None:
                    X_train, X_test_file, Ep_train, Ep_test_file, ecal_train, ecal_test_file = data_obj.data_preparation(params)
                else:
                    dw_train = data_weights[file_no]
                    ind_train = indices_train_list[file_no]
                    ind_test = indices_test_list[file_no]
                    X_train, X_test_file, Ep_train, Ep_test_file, ecal_train, ecal_test_file = data_obj.data_preparation(params, dw_train, ind_train, ind_test)
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
                data_obj.empty_datavars()
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
                    generated_images, gen_aux, gen_ecal = self.gen.generate(num=batch_size, epoch=epoch)

                    d_loss = self.disc.train_step(params, optimizer_d, image_batch, energy_batch, ecal_batch, \
                                                    generated_images, gen_aux, gen_ecal)
                    epoch_disc_loss.append(d_loss)  

                    g_loss = self.gen.train_step(params, optimizer_g, batch_size, epoch, self.disc)

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
                generated_images, gen_aux, gen_ecal = self.gen.generate(num=batch_size, epoch=epoch)

                d_test_loss = self.disc.test_step(params, image_batch, energy_batch, ecal_batch, \
                                                    generated_images, gen_aux, gen_ecal)
                disc_test_loss_list.append(d_test_loss)

                gen_test_loss = self.gen.test_step(params, batch_size, epoch, self.disc)
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

            # mem_dict = tf.config.experimental.get_memory_info('GPU:0')
            # print('Memory usage after the testing part: {}'.format(mem_dict['current']))

            del X_train
            del X_test
            del X_test_file
            del Ep_train
            del Ep_test
            del Ep_test_file
            del ecal_train
            del ecal_test
            del ecal_test_file

            # mem_dict = tf.config.experimental.get_memory_info('GPU:0')
            # print('Memory usage after deleting all data: {}'.format(mem_dict['current']))
            
            # Call validation function
                # should return validation results
            if data_weights is None:
                validation_metric = validate(self.gen, percent=params['percent'], keras_dformat=params['keras_dformat'], data_path=params['data_dir'])
            else:
                validation_metric = validate_weighted(self.gen, params['data_dir'], data_obj.filenames, data_weights, indices_train_list, params['percent'], params['keras_dformat'])
            
            # Append epoch losses to dictionary
            train_history['generator'].append(generator_train_loss) # [total, True/Fake, aux, ecal]
            train_history['discriminator'].append(discriminator_train_loss)
            test_history['generator'].append(generator_test_loss)
            test_history['discriminator'].append(discriminator_test_loss)
            # Append validation values to dictionary
            train_history['validation'].append(validation_metric) # [total, metricp, metrice]
            # Append times per epoch to dictionary
            train_history['time4epoch'].append(t_epoch)
            
            # Save the hist dictionary
            pickle.dump({'train': train_history, 'test': test_history}, open(params['results_dir'] + '/TrainHist/histdict_{}.pkl'.format(gan_no), 'wb'))

            # Save weights
            self.gen.gen.save_weights(params['results_dir']+"/Weights/gen/gen_{}/params_generator_epoch_{}.hdf5".format(gan_no, epoch),overwrite=True)
            self.disc.disc.save_weights(params['results_dir']+"/Weights/disc/disc_{}/params_discriminator_epoch_{}.hdf5".format(gan_no, epoch),overwrite=True)

            # Print loss table
            loss_table(params, train_history, test_history, params['results_dir'], epoch, validation_metric[2], save=True, time4epoch = t_epoch)  
       

        # Choose the best model
        print(len(train_history['validation']))
        best_index, best_val = self._select_best(params, train_history['validation'])
        print("Best Weights Epoch: ", best_index+1)
        # Load the weights of the best model into self.gan
        weights_path = params['results_dir'] + '/Weights/gen/gen_{}/params_generator_epoch_{}.hdf5'.format(gan_no, best_index+1)
        self.gen.gen.load_weights(weights_path)
        return best_index+1, best_val


    def _select_best(self, params, validation_metric):
        if params['crit_best'] == 'total':
            val_ind = 0
        elif params['crit_best'] == 'metricp':
            val_ind = 1
        elif params['crit_best'] == 'metrice':
            val_ind = 2
        else:
            assert False, 'Unknown criterion for best epoch.'
        vals = []
        for e in range(len(validation_metric)):
            vals.append(validation_metric[e][val_ind])
        best_val = np.min(vals)
        return vals.index(best_val), best_val


###############################################################################################################


class Mixdisc(object):
    def __enter__(self):
        # tf.keras.backend.clear_session()
        # tf.keras.backend.clear_session()
        # gc.collect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Cleaning the whole default Graph
        # tf.keras.backend.clear_session()
        # gc.collect()
        logging.debug('Cleaning the graph...')
        logging.debug('Closing the session...')

    def __init__(self, params, steps_made, gens_bestepoch):
        """ Initial setup
        """
        tf.keras.backend.clear_session()
        gc.collect()
        self.disc = Discriminator(params)
        self.gens_list = []
        # Initialize a generator and load best weights for all steps_made
        for i in range(steps_made):
            self.gens_list.append(Generator(params, arch_overwrite="FloG1"))
            self.gens_list[i].load_w_epoch(params, i, gens_bestepoch[i])
        logging.debug('MIXDISC object initialized.')
        return



