# Useful functions, validation

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import numpy.core.umath_tests as umath
from tensorflow.python.framework.op_def_library import _SatisfiesTypeConstraint
from tensorflow.python.util.tf_stack import StackTraceFilter


# TO DO: check_paths, check_paramvals

def create_resultdir(params, print_outputs=False):  # DONE
    """ Prepare folders for weigths and results.
    """
    dirlist = ['', '/Weights/', '/Weights/disc/', '/Weights/gen/', '/Weights/mixdisc/', '/TrainHist/', '/Info/', '/DataWeights/']
    for i in range(params['num_generators']):
        ending = '_{}/'.format(i)
        dirlist.extend(['/Weights/disc/disc' + ending, '/Weights/gen/gen' + ending, '/Weights/mixdisc/mixdisc' + ending])
    for dir_name in dirlist:
        try:
            # Create target Directory
            os.mkdir(params['results_dir'] + dir_name)
            if print_outputs == True:
                print("Directory " , dir_name ,  " created ") 
        except FileExistsError:
            if print_outputs == True:
                print("Directory " , dir_name ,  " already exists")
    print('Output directories created.')
    return

#################################################################################################################################
# PLOTS AND TABLES

def plot_loss(train_history, test_history, save_folder, save=False):      #plot the losses as graph
    #generator train loss
    gen_loss=[]
    gen_generation_loss=[]
    gen_auxiliary_loss=[]
    gen_lambda5_loss=[]
    x=[]
    for epoch in range(len(train_history["generator"])):
        x.append(epoch+1)
        gen_loss.append(train_history["generator"][epoch][0])
        gen_generation_loss.append(train_history["generator"][epoch][1])
        gen_auxiliary_loss.append(train_history["generator"][epoch][2])
        gen_lambda5_loss.append(train_history["generator"][epoch][3])

    #generator test loss
    gen_test_loss=[]
    gen_test_generation_loss=[]
    gen_test_auxiliary_loss=[]
    gen_test_lambda5_loss=[]
    for epoch in range(len(test_history["generator"])):
        gen_test_loss.append(test_history["generator"][epoch][0])
        gen_test_generation_loss.append(test_history["generator"][epoch][1])
        gen_test_auxiliary_loss.append(test_history["generator"][epoch][2])
        gen_test_lambda5_loss.append(test_history["generator"][epoch][3])


    #discriminator train loss
    disc_loss=[]
    disc_generation_loss=[]
    disc_auxiliary_loss=[]
    disc_lambda5_loss=[]
    x=[]
    for epoch in range(len(train_history["discriminator"])):
        x.append(epoch+1)
        disc_loss.append(train_history["discriminator"][epoch][0])
        disc_generation_loss.append(train_history["discriminator"][epoch][1])
        disc_auxiliary_loss.append(train_history["discriminator"][epoch][2])
        disc_lambda5_loss.append(train_history["discriminator"][epoch][3])

    #discriminator test loss
    disc_test_loss=[]
    disc_test_generation_loss=[]
    disc_test_auxiliary_loss=[]
    disc_test_lambda5_loss=[]
    for epoch in range(len(test_history["discriminator"])):
        disc_test_loss.append(test_history["discriminator"][epoch][0])
        disc_test_generation_loss.append(test_history["discriminator"][epoch][1])
        disc_test_auxiliary_loss.append(test_history["discriminator"][epoch][2])
        disc_test_lambda5_loss.append(test_history["discriminator"][epoch][3])    

    
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
    plt.plot(x, gen_lambda5_loss, label = "Gen ECAL", color ="aquamarine")
    
    #plt.plot(x, disc_loss, label = "Discriminator Train", color ="red")
    plt.plot(x, disc_generation_loss, label = "Disc True/Fake", color ="red")
    plt.plot(x, disc_auxiliary_loss, label = "Disc AUX", color ="orange")
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


def plot_images(image_tensor, epoch, save_folder, save=False, number=1):    #plot images of trainingsdata or generator
    xx = np.linspace(1,25,25)
    yy = np.linspace(1,25,25)
    XX, YY = np.meshgrid(xx, yy)
    
    for i in range(number):#len(image_tensor)):
        dat=image_tensor[i]
        #print(dat.shape)
        ZZ =dat[:][:][13]
        #print(ZZ.shape)

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(YY, XX, ZZ, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        #ax.plot_wireframe(xx, yy, ZZ)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('Energy');
        ax.set_ylim(25, 0)    #invert y axes, that the particle enters form the front side
        number_epoch = str(epoch)
        number_epoch = number_epoch.zfill(4)
        plt.title("Epoch "+number_epoch)
        if save==True:
            plt.savefig(save_folder+"/Save_Images/plot_" + number_epoch + ".png")
        # plt.show()
    return        


def loss_table(params, train_history,test_history, save_folder, epoch = 0, validation_metric = 0,  save=False, time4epoch=0):        #print the loss table during training
    print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}| {4:5s}'.format(
            'component', "total_loss", "fake/true_loss", "AUX_loss", "ECAL_loss"))
    print('-' * 65)

    ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}| {4:<5.2f}'
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
        f.write('{0:<22s} | {1:4s} | {2:15s} | {3:5s}| {4:5s}'.format('component', "total_loss", "fake/true_loss", "AUX_loss", "ECAL_loss"))
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
        e = time4epoch
        f.write('\nTime for Epoch: {:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
        f.write("\nValidation Metric: " + str(validation_metric))
        # f.write("\nGromov Wasserstein Distance: " + str(train_history['Gromov_Wasserstein_validation'][-1]))
        f.write("\n\n")
        f.close()                    
    return


####################################################################################################################
# VALIDATION

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

    results_internal = {}

    for energy in energies:
        #Relative error on mean moment value for each moment and each axis
        x_act= np.mean(var["momentX_act"+ str(energy)], axis=0) # first calculate the average of moment values over the images
        x_gan= np.mean(var["momentX_gan"+ str(energy)], axis=0)
        y_act= np.mean(var["momentY_act"+ str(energy)], axis=0)
        y_gan= np.mean(var["momentY_gan"+ str(energy)], axis=0)
        z_act= np.mean(var["momentZ_act"+ str(energy)], axis=0)
        z_gan= np.mean(var["momentZ_gan"+ str(energy)], axis=0)
        var["posx_error"+ str(energy)]= (x_act - x_gan)/x_act # relative error of the mean moment values
        var["posy_error"+ str(energy)]= (y_act - y_gan)/y_act
        var["posz_error"+ str(energy)]= (z_act - z_gan)/z_act
        #Taking absolute of errors and adding for each axis then scaling by 3
        # Average the moment errors over axes
        var["pos_error"+ str(energy)]= (np.absolute(var["posx_error"+ str(energy)]) + np.absolute(var["posy_error"+ str(energy)])+ np.absolute(var["posz_error"+ str(energy)]))/3
        #Summing over moments and dividing for number of moments
        var["pos_total"+ str(energy)]= np.sum(var["pos_error"+ str(energy)])/m
        metricp += var["pos_total"+ str(energy)]
        
        # Take profile along each axis and find mean along events
        sumxact, sumyact, sumzact = np.mean(var["sumsx_act" + str(energy)], axis=0), np.mean(var["sumsy_act" + str(energy)], axis= 0), np.mean(var["sumsz_act" + str(energy)], axis=0)
        sumxgan, sumygan, sumzgan = np.mean(var["sumsx_gan" + str(energy)], axis=0), np.mean(var["sumsy_gan" + str(energy)], axis=0), np.mean(var["sumsz_gan" + str(energy)], axis=0)
        var["eprofilex_error"+ str(energy)] = np.divide((sumxact - sumxgan), sumxact)
        var["eprofiley_error"+ str(energy)] = np.divide((sumyact - sumygan), sumyact)
        var["eprofilez_error"+ str(energy)] = np.divide((sumzact - sumzgan), sumzact)
        #Take absolute of error and mean for all events
        var["eprofilex_total"+ str(energy)]= np.sum(np.absolute(var["eprofilex_error"+ str(energy)]))/x
        var["eprofiley_total"+ str(energy)]= np.sum(np.absolute(var["eprofiley_error"+ str(energy)]))/y
        var["eprofilez_total"+ str(energy)]= np.sum(np.absolute(var["eprofilez_error"+ str(energy)]))/z

        var["eprofile_total"+ str(energy)]= (var["eprofilex_total"+ str(energy)] + var["eprofiley_total"+ str(energy)] + var["eprofilez_total"+ str(energy)])/3
        metrice += var["eprofile_total"+ str(energy)]
        if ang:
            var["angle_error"+ str(energy)] = np.mean(np.absolute((var[angtype + "_act" + str(energy)] - var[angtype + "_gan" + str(energy)])/var[angtype + "_act" + str(energy)]))
            metrica += var["angle_error"+ str(energy)]

        results_internal["posx_error"+ str(energy)] = var["posx_error"+ str(energy)]
        results_internal["posy_error"+ str(energy)] = var["posy_error"+ str(energy)]
        results_internal["posz_error"+ str(energy)] = var["posz_error"+ str(energy)]
        results_internal["pos_error"+ str(energy)] = var["pos_error"+ str(energy)] # average over axes for each moment
        results_internal["pos_total" + str(energy)] = var["pos_total"+ str(energy)] # average over moments
        results_internal["eprofilex_error"+ str(energy)] = var["eprofilex_error"+ str(energy)]
        results_internal["eprofiley_error"+ str(energy)] = var["eprofiley_error"+ str(energy)]
        results_internal["eprofilez_error"+ str(energy)] = var["eprofilez_error"+ str(energy)]
        results_internal["eprofilex_total"+ str(energy)] = var["eprofilex_total"+ str(energy)]
        results_internal["eprofiley_total"+ str(energy)] = var["eprofiley_total"+ str(energy)]
        results_internal["eprofilez_total"+ str(energy)] = var["eprofilez_total"+ str(energy)]
        results_internal["eprofile_total"+ str(energy)] = var["eprofile_total"+ str(energy)] # average over axes
        
    metricp = metricp/len(energies) # averaging over energies
    metrice = metrice/len(energies) # averaging over energies
    if ang:metrica = metrica/len(energies)
    tot = metricp + metrice
    if ang:tot = tot + metrica
    result = [tot, metricp, metrice]
    if ang: result.append(metrica)

    return result, results_internal


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
      var["ecal_gan"+ str(energy)] = np.sum(var["events_gan" + str(energy)], axis = (1, 2, 3))
      var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)] = get_sums(var["events_act" + str(energy)])
      var["sumsx_gan"+ str(energy)], var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)] = get_sums(var["events_gan" + str(energy)])
      var["momentX_act" + str(energy)], var["momentY_act" + str(energy)], var["momentZ_act" + str(energy)]= get_moments(var["sumsx_act"+ str(energy)], var["sumsy_act"+ str(energy)], var["sumsz_act"+ str(energy)], var["ecal_act"+ str(energy)], m, x=x, y=y, z=z)
      var["momentX_gan" + str(energy)], var["momentY_gan" + str(energy)], var["momentZ_gan" + str(energy)] = get_moments(var["sumsx_gan"+ str(energy)], var["sumsy_gan"+ str(energy)], var["sumsz_gan"+ str(energy)], var["ecal_gan"+ str(energy)], m, x=x, y=y, z=z)
      if ang: var["angle_gan"+ str(energy)]= measPython(var["events_gan" + str(energy)])
    return metric(var, energies, m, angtype='angle', x=x, y=y, z=z, ang=ang)

#validation script
#keras_dformat= "channels_first"
def validate_extended(generator, data_path, percent=20, keras_dformat='channels_first'):
    X=np.zeros((1,25,25,25))
    y=np.zeros((1))
    file = h5py.File(data_path + "EleEscan_1_2.h5",'r')   #file_1 does not work and gives nan values
    e_file = file.get('target')               #Target ist die Zielenergie, entweder E_p oder Ecal
    X_file = np.array(file.get('ECAL'))       #ECAL ist die 3D Energieverteilung/das Bild
    y_file = np.array(e_file[:,1])
    file.close()
    file2 = h5py.File(data_path + "EleEscan_1_3.h5",'r')   #file_1 does not work and gives nan values
    e_file2 = file2.get('target')               #Target ist die Zielenergie, entweder E_p oder Ecal - the target is the total energy E_p or Ecal
    X_file2 = np.array(file2.get('ECAL'))       #ECAL ist die 3D Energieverteilung/das Bild - ECAL is the 3D energy distribution (3D image)
    y_file2 = np.array(e_file2[:,1])
    file2.close()
    X = np.concatenate((X_file, X_file2))
    y = np.concatenate((y_file, y_file2))

    X[X < 1e-6] = 0  #remove unphysical values

    X = np.delete(X, 0,0)   #heißt Lösche Element 0 von Spalte 0 - remove element 0 from column 0
    y = np.delete(y, 0,0)   #heißt Lösche Element 0 von Spalte 0
    
    X_val = X
    y_val = y

    X_val=X_val[:int(len(X_val)*percent/100),:]
    y_val=y_val[:int(len(y_val)*percent/100)]

    # tensorflow ordering
    X_val = np.expand_dims(X_val, axis=-1)


    if keras_dformat !='channels_last':
        X_val = np.moveaxis(X_val, -1,1)

    y_val=y_val/100

    nb_val = X_val.shape[0]


    X_val = X_val.astype(np.float32)
    y_val = y_val.astype(np.float32)
    if keras_dformat =='channels_last':
        ecal_val = np.sum(X_val, axis=(1, 2, 3))
    else:
        ecal_val = np.sum(X_val, axis=(2, 3, 4))

    X_val = np.squeeze(X_val)
    var={}
    tolerance = 5
    energies = [0, 50, 100, 200, 250, 300, 400, 500]
    data0 = X_val  #the generated data
    data1 = y_val    #aux
    ecal = ecal_val
    ang=0
    for energy in energies:
        if energy==0:
            var["events_act" + str(energy)]=data0
            var["energy" + str(energy)]=data1
            if ang: var["angle_act" + str(energy)]=data[2]
            var["ecal_act" + str(energy)]=ecal
            var["index" + str(energy)] = var["events_act" + str(energy)].shape[0]
        else:
            var["indexes" + str(energy)] = np.where((data1 > (energy - tolerance)/100. ) & ( data1 < (energy + tolerance)/100.))
            var["events_act" + str(energy)]=data0[var["indexes" + str(energy)]]
            var["energy" + str(energy)]=data1[var["indexes" + str(energy)]]
            if ang:  var["angle_act" + str(energy)]=data[2][var["indexes" + str(energy)]]
            var["ecal_act" + str(energy)]=ecal[var["indexes" + str(energy)]]
            var["index" + str(energy)] = var["events_act" + str(energy)].shape[0]


    #validation

    #var = sortEnergy([np.squeeze(X_test), Y_test], np.squeeze(ecal_test), energies, ang=0)
    nb_test = len(y_val); latent_size =200
    noise = np.random.normal(0.1, 1, (nb_test, latent_size))
    generator_ip = np.multiply(data1.reshape((-1, 1)), noise) # using the same set of E_p for the generated data

    #sess = tf.compat.v1.Session(graph = infer_graph)
    generated_images = generator.predict(generator_ip,batch_size=128)
    #generated_images = sess.run(l_output, feed_dict = {l_input:generator_ip})
    generated_images= np.squeeze(generated_images)

    #generated_images = generator.predict(generator_ip, verbose=False, batch_size=batch_size)
    result, results_internal = OptAnalysisShort(var, generated_images, energies, ang=0)
    print('Analysing............')
    # All of the results correspond to mean relative errors on different quantities
    print('Result = ', result[1]) #optimize over result[0]
    #print("Value to optimize: ", np.round(result[1],3))
    #pickle.dump({'results': analysis_history}, open(resultfile, 'wb'))
    return result, results_internal


def validate(generator, data_path, percent=20, keras_dformat='channels_first'):
    X_val=np.zeros((1,25,25,25))
    y_val=np.zeros((1))
    file = h5py.File(data_path + "EleEscan_1_2.h5",'r')   #file_1 does not work and gives nan values
    e_file = file.get('target')               #Target ist die Zielenergie, entweder E_p oder Ecal
    X_file = np.array(file.get('ECAL'))       #ECAL ist die 3D Energieverteilung/das Bild
    y_file = np.array(e_file[:,1])
    file.close()
    file2 = h5py.File(data_path + "EleEscan_1_3.h5",'r')   #file_1 does not work and gives nan values
    e_file2 = file2.get('target')               #Target ist die Zielenergie, entweder E_p oder Ecal - the target is the total energy E_p or Ecal
    X_file2 = np.array(file2.get('ECAL'))       #ECAL ist die 3D Energieverteilung/das Bild - ECAL is the 3D energy distribution (3D image)
    y_file2 = np.array(e_file2[:,1])
    file2.close()
    X_val = np.concatenate((X_file, X_file2))
    y_val = np.concatenate((y_file, y_file2))

    X_val[X_val < 1e-6] = 0  #remove unphysical values

    X_val = np.delete(X_val, 0,0)   #heißt Lösche Element 0 von Spalte 0 - remove element 0 from column 0
    y_val = np.delete(y_val, 0,0)   #heißt Lösche Element 0 von Spalte 0

    X_val=X_val[:int(len(X_val)*percent/100),:]
    y_val=y_val[:int(len(y_val)*percent/100)]

    # tensorflow ordering
    X_val = np.expand_dims(X_val, axis=-1)


    if keras_dformat !='channels_last':
        X_val = np.moveaxis(X_val, -1,1)

    y_val=y_val/100

    nb_val = X_val.shape[0]


    X_val = X_val.astype(np.float32)
    y_val = y_val.astype(np.float32)
    if keras_dformat =='channels_last':
        ecal_val = np.sum(X_val, axis=(1, 2, 3))
    else:
        ecal_val = np.sum(X_val, axis=(2, 3, 4))

    X_val = np.squeeze(X_val)
    var={}
    tolerance = 5
    energies = [0, 50, 100, 200, 250, 300, 400, 500]
    # data0 = X_val  #the generated data
    # data1 = y_val    #aux
    ecal = ecal_val
    ang=0
    for energy in energies:
        if energy==0:
            var["events_act" + str(energy)]=X_val
            var["energy" + str(energy)]=y_val
            if ang: var["angle_act" + str(energy)]=data[2]
            var["ecal_act" + str(energy)]=ecal
            var["index" + str(energy)] = var["events_act" + str(energy)].shape[0]
        else:
            var["indexes" + str(energy)] = np.where((y_val > (energy - tolerance)/100. ) & ( y_val < (energy + tolerance)/100.))
            var["events_act" + str(energy)]=X_val[var["indexes" + str(energy)]]
            var["energy" + str(energy)]=y_val[var["indexes" + str(energy)]]
            if ang:  var["angle_act" + str(energy)]=data[2][var["indexes" + str(energy)]]
            var["ecal_act" + str(energy)]=ecal[var["indexes" + str(energy)]]
            var["index" + str(energy)] = var["events_act" + str(energy)].shape[0]

    #validation

    #var = sortEnergy([np.squeeze(X_test), Y_test], np.squeeze(ecal_test), energies, ang=0)
    nb_test = len(y_val); latent_size =200
    noise = np.random.normal(0.1, 1, (nb_test, latent_size))
    generator_ip = np.multiply(y_val.reshape((-1, 1)), noise) # using the same set of E_p for the generated data

    X_val = None
    y_val = None
    ecal = None
    ecal_val = None

    # del data0
    # del data1
    # del X
    # del y
    # del X_val
    # del y_val
    # del ecal
    # del ecal_val

    #sess = tf.compat.v1.Session(graph = infer_graph)
    generated_images = generator.generate(num = nb_test, generator_input = generator_ip)
    #generated_images = sess.run(l_output, feed_dict = {l_input:generator_ip})
    generated_images= np.squeeze(generated_images)

    #generated_images = generator.predict(generator_ip, verbose=False, batch_size=batch_size)
    result, _ = OptAnalysisShort(var, generated_images, energies, ang=0)
    print('Analysing............')
    # All of the results correspond to mean relative errors on different quantities
    print('Result = ', result[2]) #optimize over result[0]
    #print("Value to optimize: ", np.round(result[1],3))
    #pickle.dump({'results': analysis_history}, open(resultfile, 'wb'))
    return result   # [tot, metricp, metrice]


def validate_weighted(generator, data_path, data_files_list, data_weights, ind_train_list, percent=20, keras_dformat='channels_first'):
    X_val=np.zeros((1,25,25,25))
    y_val=np.zeros((1))
    filename1 = "EleEscan_1_2.h5"
    file = h5py.File(data_path + "EleEscan_1_2.h5",'r')   #file_1 does not work and gives nan values
    e_file = file.get('target')               #Target ist die Zielenergie, entweder E_p oder Ecal
    X_file = np.array(file.get('ECAL'))       #ECAL ist die 3D Energieverteilung/das Bild
    y_file = np.array(e_file[:,1])
    file.close()
    filename2 = "EleEscan_1_3.h5"
    file2 = h5py.File(data_path + "EleEscan_1_3.h5",'r')   #file_1 does not work and gives nan values
    e_file2 = file2.get('target')               #Target ist die Zielenergie, entweder E_p oder Ecal - the target is the total energy E_p or Ecal
    X_file2 = np.array(file2.get('ECAL'))       #ECAL ist die 3D Energieverteilung/das Bild - ECAL is the 3D energy distribution (3D image)
    y_file2 = np.array(e_file2[:,1])
    file2.close()
    X_val = np.concatenate((X_file, X_file2))
    y_val = np.concatenate((y_file, y_file2))

    X_val[X_val < 1e-6] = 0  #remove unphysical values

    X_val = np.delete(X_val, 0,0)   #heißt Lösche Element 0 von Spalte 0 - remove element 0 from column 0
    y_val = np.delete(y_val, 0,0)   #heißt Lösche Element 0 von Spalte 0

    # X_val=X_val[:int(len(X_val)*percent/100),:]   # maybe return this possibility back
    # y_val=y_val[:int(len(y_val)*percent/100)]

    # tensorflow ordering
    X_val = np.expand_dims(X_val, axis=-1)


    if keras_dformat !='channels_last':
        X_val = np.moveaxis(X_val, -1,1)

    y_val=y_val/100

    nb_val = X_val.shape[0]


    X_val = X_val.astype(np.float32)
    y_val = y_val.astype(np.float32)
    if keras_dformat =='channels_last':
        ecal_val = np.sum(X_val, axis=(1, 2, 3))
    else:
        ecal_val = np.sum(X_val, axis=(2, 3, 4))

    X_val = np.squeeze(X_val)

    # Selection using weights
    files_inds = [idx for idx in range(len(data_files_list)) if data_files_list[idx] == filename1]  # get index of the file with name filename1
    files_inds.extend([idx for idx in range(len(data_files_list)) if data_files_list[idx] == filename2])
    file_weights = []   # weight of data in files filename1, filename2
    for ind in files_inds:
        file_weights.extend(data_weights[ind])
    file_weights = file_weights/np.sum(file_weights) # normalize weights to sum to 1, weights from both files are stored in one list
    num_weighted = int(nb_val*0.7)
    data_val_indices = np.random.choice(len(file_weights), num_weighted, replace=True, p=file_weights)  # randomly select 70 % of the data using weights
    X_val = X_val[data_val_indices]
    y_val = y_val[data_val_indices]
    ecal_val = ecal_val[data_val_indices]

    var={}
    tolerance = 5
    energies = [0, 50, 100, 200, 250, 300, 400, 500]
    # data0 = X_val  #the generated data
    # data1 = y_val    #aux
    ecal = ecal_val
    ang=0
    for energy in energies:
        if energy==0:
            var["events_act" + str(energy)]=X_val
            var["energy" + str(energy)]=y_val
            if ang: var["angle_act" + str(energy)]=data[2]
            var["ecal_act" + str(energy)]=ecal
            var["index" + str(energy)] = var["events_act" + str(energy)].shape[0]
        else:
            var["indexes" + str(energy)] = np.where((y_val > (energy - tolerance)/100. ) & ( y_val < (energy + tolerance)/100.))
            var["events_act" + str(energy)]=X_val[var["indexes" + str(energy)]]
            var["energy" + str(energy)]=y_val[var["indexes" + str(energy)]]
            if ang:  var["angle_act" + str(energy)]=data[2][var["indexes" + str(energy)]]
            var["ecal_act" + str(energy)]=ecal[var["indexes" + str(energy)]]
            var["index" + str(energy)] = var["events_act" + str(energy)].shape[0]

    #validation

    #var = sortEnergy([np.squeeze(X_test), Y_test], np.squeeze(ecal_test), energies, ang=0)
    nb_test = len(y_val); latent_size =200
    noise = np.random.normal(0.1, 1, (nb_test, latent_size))
    generator_ip = np.multiply(y_val.reshape((-1, 1)), noise) # using the same set of E_p for the generated data

    X_val = None
    y_val = None
    ecal = None
    ecal_val = None

    # del data0
    # del data1
    # del X
    # del y
    # del X_val
    # del y_val
    # del ecal
    # del ecal_val

    #sess = tf.compat.v1.Session(graph = infer_graph)
    generated_images = generator.generate(num = num_weighted, generator_input = generator_ip)
    #generated_images = sess.run(l_output, feed_dict = {l_input:generator_ip})
    generated_images= np.squeeze(generated_images)

    #generated_images = generator.predict(generator_ip, verbose=False, batch_size=batch_size)
    result, _ = OptAnalysisShort(var, generated_images, energies, ang=0)
    print('Analysing............')
    # All of the results correspond to mean relative errors on different quantities
    print('Result = ', result[2]) #optimize over result[0]
    #print("Value to optimize: ", np.round(result[1],3))
    #pickle.dump({'results': analysis_history}, open(resultfile, 'wb'))
    return result   # [tot, metricp, metrice]



def check_paramvals(params):
    """ Check if all parameters have valid values
    """

    print('params: valid.')
    return


def check_paths():
    """ Check data path - if it exists and is not empty
    """
    return




