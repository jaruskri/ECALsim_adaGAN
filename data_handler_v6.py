
import os
import logging
import numpy as np
import h5py
from sklearn.model_selection import train_test_split


class DataParent(object):
    def __init__(self, params):
        # Copy key params, list available files
        self.num_samples = None
        self.folder = params['data_dir']
        self.dataset = params['dataset']
        self.filenames = os.listdir(self.folder)

        # Adjust the number fo training files
        logging.debug('Total number of training data files: {}'.format(len(self.filenames)))
        if isinstance(params['nb_train_files'], str):
            logging.debug('Using all training data files.')
        else:
            if params['nb_train_files']>len(self.filenames):
                logging.debug('Less training data files than requested, using all {} files available.'.format(len(self.filenames)))
            else:
                self.filenames = self.filenames[:params['nb_train_files']]
                logging.debug('Working with {} training data files.'.format(params['nb_train_files']))
        
        # Prepare variables
        self.data_shape = None
        self.X = None
        self.Ep = None
        self.is_angle = (params['dataset'] == 'angle')
        if self.is_angle:
            self.angle = None
        return

    def get_shape_num(self, params):
        if params['data_load2mem'] and (not self.X is None):
            temp = np.shape(self.X)
            self.data_shape = temp[1:]
            self.num_samples = temp[0]
        elif params['data_load2mem'] and (self.X is None):
            logging.info("self.X is None, loading data from ... {}".format(params['data_dir']))
            self.load(params)
            self.get_shape_num(params)
        else:
            logging.debug('Dataset too large to be loaded at once. Going through data files and counting ... {}'.format(params['data_dir']))
            self.num_samples = 0
            for file_no in range(len(self.filenames)):
                self.load_file(params, file_no)
                # print(np.shape(self.X)[0])
                # print(np.shape(self.X))
                self.num_samples += np.shape(self.X)[0]
            self.data_shape = np.shape(self.X)[1:]
            self.empty_datavars()
        logging.info('Number of samples in a dataset: {}'.format(self.num_samples))
        logging.info('Data shape: {}'.format(self.data_shape))
        return

    def empty_datavars(self):
        # logging.info('Emptying data variables X, Ep (, angle).')
        self.X, self.Ep = None, None
        if self.is_angle:
            self.angle = None
        return

    def load(self, params):
        """ Loading all training data into memory, if requested.
        """
        raise NotImplementedError("Not implemented in the parent class.")
        # assert False, 'DataParent base class has no load method defined.'

    def load_file(self, params, append=False, get_num=False):
        """ Load a specific file into self.X, self.Ep, self.angle
        """
        raise NotImplementedError("Not implemented in the parent class.")
        # assert False, 'DataParent base class has no load_file method defined.'

    def data_preparation(self, params):
        """ Train test split of the data
        """
        raise NotImplementedError("Not implemented in the parent class.")
        # assert False, 'DataParent base class has no data_preparation method defined.'


class EnergData(DataParent):
    """ Class to manipulate with the energy-type dataset
        Inherits methods from DataParent class
    """
    def load(self, params):
        if params['data_load2mem']:
            n_files = len(self.filenames)
            for i in n_files:
                self.load_file(params, self.filenames[i], True)
                logging.info('Loaded: {}/{}.'.format(i+1, n_files))
        else:
            logging.info('Loading file-by-file was requested, use the load_file method.')
        return

    def load_file(self, params, fileno, append=False, get_num=False):
        filename = params['data_dir'] + self.filenames[fileno]
        # print(filename)
        f = h5py.File(filename, 'r')
        x = np.array(f.get('ECAL'))
        ep = np.array(f.get('target')[:,1])
        f.close()
        x[x < 1e-6] = 0                                 # removing unphysical values
        if append and (not self.X is None):
            np.concatenate((self.X, x), axis=0)
            np.concatenate((self.Ep, ep), axis=0)
        else:
            self.X = x
            self.Ep = ep
        if get_num:
            if self.num_samples is None: self.num_samples, self.data_shape = self.X.shape[0], self.X.shape[1:]
            self.num_samples += self.X.shape[0]
        return

    def data_preparation(self, params, data_weights = None, ind_train = None, ind_test = None):
        """ Train-test splitting
        """
        # Trial run
        if params['trial_run']:
            nb = int(self.X.shape[0]*0.1)
            self.X = self.X[:nb,:,:,:]
            self.Ep = self.Ep[:nb]
        
        if data_weights is None:
            X_train, X_test, Ep_train, Ep_test = train_test_split(self.X, self.Ep, train_size = 1-params['test_data_portion'], \
                                                    test_size = params['test_data_portion'])
        else:
            w_train = np.array(data_weights)
            X_train = self.X[ind_train]
            X_test = self.X[ind_test]
            Ep_train = self.Ep[ind_train]
            Ep_test = self.Ep[ind_test]


        # Tensorflow ordering
        X_train =np.expand_dims(X_train, axis=-1)  #macht jeden Eintrag in der Liste zu einer Unterliste [1,2,3]->[[1],[2],[3]]
        X_test = np.expand_dims(X_test, axis=-1)

        # "channels ordering" (keras_dformat)
        if params['keras_dformat'] != 'channels_last':
            X_train =np.moveaxis(X_train, -1, 1)    #Dreht die Matrix, damit die Dimension passt
            X_test = np.moveaxis(X_test, -1,1)

        # Normalization (originaly in 2-500 Gev, now 20-5000 MeV) - WHY?
        Ep_train = Ep_train/100
        Ep_test = Ep_test/100
        
        # Check count for batch size
        nb_train = X_train.shape[0]
        if nb_train < params['batch_size']:
            print("\nERROR: batch_size is larger than trainings data")
            print("batch_size: ", params['batch_size'])
            print("trainings data: ", nb_train, "\n")
        # assert nb_train >= params['batch_size'], 'batch_size is larger than training data'

        # Convert to float32
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        Ep_train = Ep_train.astype(np.float32)
        Ep_test = Ep_test.astype(np.float32)

        # Ep - set correct dimensions
        Ep_train = np.expand_dims(Ep_train, axis=-1)
        Ep_test = np.expand_dims(Ep_test, axis=-1)

        # Prepare ECAL (sum of deposited energies)
        if params['keras_dformat'] =='channels_last':
            ecal_train = np.sum(X_train, axis=(1, 2, 3))
            ecal_test = np.sum(X_test, axis=(1, 2, 3))
        else:
            ecal_train = np.sum(X_train, axis=(2, 3, 4))
            ecal_test = np.sum(X_test, axis=(2, 3, 4))

        logging.debug('Data preparation ... done')

        if data_weights is None:
            return X_train, X_test, Ep_train, Ep_test, ecal_train, ecal_test
        else:
            return X_train, X_test, Ep_train, Ep_test, ecal_train, ecal_test

    
    def prepare_images_w(self, params):
        """ Prepare only the image data
        """
        # Trial run
        if params['trial_run']:
            nb = int(self.X.shape[0]*0.1)
            self.X = self.X[:nb,:,:,:]
            self.Ep = self.Ep[:nb]

        indices_full = list(range(np.shape(self.X)[0])) # list of indices for all data
        X_train, _, ind_train, ind_test = train_test_split(self.X, indices_full, train_size = 1-params['test_data_portion'], \
                                                test_size = params['test_data_portion'])

        # Tensorflow ordering
        X_train =np.expand_dims(X_train, axis=-1)  #macht jeden Eintrag in der Liste zu einer Unterliste [1,2,3]->[[1],[2],[3]]

        # "channels ordering" (keras_dformat)
        if params['keras_dformat'] != 'channels_last':
            X_train =np.moveaxis(X_train, -1, 1)    
        
        # Check count for batch size
        nb_train = X_train.shape[0]
        assert nb_train >= params['batch_size'], 'batch_size is larger than training data'

        # Convert to float32
        X_train = X_train.astype(np.float32)

        return X_train, ind_train, ind_test



# class AngleData(DataParent):
#     """ Class to manipulate with the angle-type dataset
#         Inherits methods from DataParent class
#     """