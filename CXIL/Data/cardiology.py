

import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pickle
from scipy.signal import resample
import random

def remove_patients_with_empty_frames(ecg_frames, ecg_labels):
    patients_with_empty_frames = [name for name,frames in ecg_frames.items() if np.array(frames).shape[0] == 0]

    if ecg_frames is not None:
        [ecg_frames.pop(key) for key in patients_with_empty_frames]
        [ecg_labels.pop(key) for key in patients_with_empty_frames]
    return ecg_frames, ecg_labels
    
#    if ppg_frames is not None:
#        [ppg_frames.pop(key) for key in patients_with_empty_frames]
#        [ppg_labels.pop(key) for key in patients_with_empty_frames]

#def obtain_train_test_split(ecg_frames):
#    """ Split Patients Into Train, Val, and Test """
#    patient_numbers = list(ecg_frames.keys())
#    
#    """ Obtain Test Set """
#    test_ratio = 0.2
#    test_length = int(len(patient_numbers)*test_ratio)
#    random.seed(0)
#    patient_numbers_test = random.sample(patient_numbers,test_length)
#    patient_numbers_train = list(set(patient_numbers) - set(patient_numbers_test))
#    
#    """ Obtain Train and Test Split """
#    val_ratio = 0.2
#    val_length = int(len(patient_numbers_train)*val_ratio)
#    random.seed(0)
#    patient_numbers_val = random.sample(patient_numbers_train,val_length)
#    patient_numbers_train = list(set(patient_numbers_train) - set(patient_numbers_val))
#    
#    return patient_numbers_train,patient_numbers_val,patient_numbers_test

def generate_cardiology():
    #https://github.com/danikiyasseh/loading-physiological-data/blob/master/load_cardiology_ecg.py
    #TODO Cardiology Data not openly available - rather use Phsiology data 
    #https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6784839/
    basepath='./XIL/Data/data/cardiology'
    if not os.path.exists(f'./XIL/Data/data/cardiology'):
        #Download data 
        pass
    s = 200
    original_frame_length = 6000
    samples_per_frame = 256
    resampled_length = 2500
    """ Determine Number of Frames Per Original Frame """
    nframes = original_frame_length//samples_per_frame
    samples_to_take_per_frame = samples_per_frame*nframes

    """ All Files in Directory """
    files = os.listdir(basepath)
    """ Return Unique Patient Ids """
    unique_patient_numbers = np.unique([file.split('_')[0] for file in files if not os.path.isdir(os.path.join(basepath,file))])

    classification = 'all' #all

    inputs = dict()
    outputs = dict()
    all_labels = []
    for patient_number in unique_patient_numbers:
        inputs[patient_number] = []
        outputs[patient_number] = []

        """ Load Frame Data """
        filename = [file for file in files if patient_number in file and 'ecg' in file][0]
        f = open(os.path.join(basepath,filename),'rb')
        frame = np.fromfile(f,dtype=np.int16) #6000x1
        
        """ Load Group Label File """    
        group_label = [file for file in files if patient_number in file and 'grp' in file][0]
        with open(os.path.join(basepath,group_label)) as json_file:
            data = json.load(json_file)
        
        onsets = [episode['onset']-1 for episode in data['episodes']] #=1 for python start at 0
        offsets = [episode['offset'] for episode in data['episodes']]
        rhythms = [episode['rhythm_name'] for episode in data['episodes']]
        
        for nframe in range(nframes):
            start_sample = nframe * samples_per_frame
            end_sample = start_sample + samples_per_frame
            mini_frame = frame[start_sample:end_sample]
            for i in range(len(rhythms)):
                if onsets[i] <= start_sample < offsets[i]:
                    mini_label = rhythms[i]     
                    if mini_label == 'AVB_TYPE2':
                        mini_label = 'AVB'
                    elif mini_label == 'AFL':
                        mini_label = 'AFIB'
                    elif mini_label == 'SUDDEN_BRADY':
                        break
            
            if mini_label == 'SUDDEN_BRADY': #dont record sudden brady
                continue
            
            """ Resample Frame """
            mini_frame = resample(mini_frame,resampled_length)
            
            #""" Binarize Labels """
            #if classification == 'binary':
            #    if mini_label == 'NSR':
            #        mini_label = 0
            #    else:
            #        mini_label = 1
            
            all_labels.append(mini_label)
            inputs[patient_number].append(mini_frame)
            outputs[patient_number].append(mini_label)
        
        
        inputs[patient_number] = np.array(inputs[patient_number])
        outputs[patient_number] = np.array(outputs[patient_number])
    """ Retrieve Unique Class Names """
    unique_labels = []
    for label in all_labels:
        if label not in unique_labels:
            unique_labels.append(label)

    """ Convert Drug Names to Labels """
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(unique_labels)
    for patient_number,labels in outputs.items():
        outputs[patient_number] = label_encoder.transform(labels)

    inputs, outputs = remove_patients_with_empty_frames(inputs, outputs)
    #""" Make New Directory to Avoid Contamination """
    #savepath = os.path.join(basepath,'patient_data','%s_classes' % classification)
    #try:
    #    os.chdir(savepath)
    #except:
    #    os.makedirs(savepath)
    #""" Save Inputs and Labels Dicts For Splitting Later """
    with open(os.path.join('./XIL/Data/data/cardiology','ecg_signal_frames_cardiology.pkl'),'wb') as f:
        pickle.dump(inputs,f)
    with open(os.path.join('./XIL/Data/data/cardiology','ecg_signal_arrhythmia_labels_cardiology.pkl'),'wb') as f:
        pickle.dump(outputs,f)