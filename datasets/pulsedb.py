# ToDo
# extract pulsedb dataset loader into this file


import os
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import TensorDataset

def load_train_spectrogram():
    Signals_ECG = np.load(f"/PublicSSD/jhpark/ECGBP/1_preprocessed_data/pulse_v1/Signals_Train_spectrogram_ECG.npy")
    Signals_PPG = np.load(f"/PublicSSD/jhpark/ECGBP/1_preprocessed_data/pulse_v1/Signals_Train_spectrogram_PPG.npy")
    SBPLabels = np.load(f"/PublicSSD/jhpark/ECGBP/1_preprocessed_data/pulseDB/SBPLabels.npy")
    return Signals_ECG, Signals_PPG, SBPLabels

def load_train_small_spectrogram(channel):
    # Signals = np.load(f"/PublicSSD/jhpark/ECGBP/1_preprocessed_data/pulse_v1/Signals_Train_spectrogram_{channel}_small.npy")
    # SBPLabels = np.load(f"/PublicSSD/jhpark/ECGBP/1_preprocessed_data/pulse_v1/SBPLabels_small.npy")
    Signals = np.load(f"/PublicSSD/jhpark/ECGBP/1_preprocessed_data/pulse_v1/Signals_Train_spectrogram_{channel}_sixman.npy")
    SBPLabels = np.load(f"/PublicSSD/jhpark/ECGBP/1_preprocessed_data/pulse_v1/SBPLabels_sixman.npy")
    return Signals, SBPLabels

def load_test_spectrogram():
    Signals_ECG = np.load(f"/PublicSSD/jhpark/ECGBP/1_preprocessed_data/pulse_v1/Signals_CalBased_Test_spectrogram_ECG.npy")
    Signals_PPG = np.load(f"/PublicSSD/jhpark/ECGBP/1_preprocessed_data/pulse_v1/Signals_CalBased_Test_spectrogram_PPG.npy")
    SBPLabels = np.load(f"/PublicSSD/jhpark/ECGBP/1_preprocessed_data/pulseDB/SBPLabels_CalBased_Test.npy")
    return Signals_ECG, Signals_PPG, SBPLabels

def normalize_signal(Signals, mean, std):
    n_channel = Signals.shape[1]
    # assert n_channel == mean.shape[0] == std.shape[0]
    # ToDo: for loop is not required...
    for i in range(n_channel): 
        Signals[:,i,:] = (Signals[:,i,:] - mean) / std   
    return Signals



class PULSEDBfull(data.Dataset):
    def __init__(self, phase):
        self.phase = phase

        mean_ECG = -55.18958194
        std_ECG = 20.79336364    
        mean_PPG = -72.7312628
        std_PPG = 18.75779879
        
        if self.phase == 'train':
            Signals_ECG, Signals_PPG, self.SBPLabels = load_train_spectrogram()
            
            self.SBPLabels = self.SBPLabels.reshape(-1, 1)
            self.SBPLabels = torch.from_numpy(self.SBPLabels)

            Signals_ECG = normalize_signal(Signals_ECG, mean_ECG, std_ECG)            
            Signals_PPG = normalize_signal(Signals_PPG, mean_PPG, std_PPG)
            
            self.Signals = np.concatenate((Signals_ECG, Signals_ECG, Signals_PPG), axis=1)
            self.Signals = torch.from_numpy(self.Signals)
            
            print(f"Loading PULSEDB-train dataset")
            print(f"Signals_ECG.shape: {Signals_ECG.shape}")
            print(f"Signals_PPG.shape: {Signals_PPG.shape}")
            print(f"Signals.shape: {self.Signals.shape}")
            print(f"SBPLabels.shape: {self.SBPLabels.shape}")            
            print(f"dtype of dataset: {self.Signals.dtype}, {self.SBPLabels.dtype}")
            
            # pulsedb_dataset = TensorDataset(torch.from_numpy(Signals), torch.from_numpy(SBPLabels))
        
        if phase == 'test':
            Signals_ECG_test, Signals_PPG_test, self.SBPLabels = load_test_spectrogram()
            self.SBPLabels = self.SBPLabels.reshape(-1, 1)
            self.SBPLabels = torch.from_numpy(self.SBPLabels)

            Signals_ECG_test = normalize_signal(Signals_ECG_test, mean_ECG, std_ECG)            
            Signals_PPG_test = normalize_signal(Signals_PPG_test, mean_PPG, std_PPG)
            self.Signals = np.concatenate((Signals_ECG_test, Signals_ECG_test, Signals_PPG_test), axis=1)
            self.Signals = torch.from_numpy(self.Signals)
            
            print(f"Loading PULSEDB-CalBased-test dataset")
            print(f"Signals_ECG_test.shape: {Signals_ECG_test.shape}")
            print(f"Signals_PPG_test.shape: {Signals_PPG_test.shape}")
            print(f"SBPLabels_test.shape: {self.SBPLabels.shape}")             
            # pulsedb_dataset = TensorDataset(torch.from_numpy(Signals_ECG_test), torch.from_numpy(Signals_PPG), torch.from_numpy(SBPLabels))
                                    
        # return pulsedb_dataset
    
    def __getitem__(self, index):
        img = self.Signals[index,:]
        target = self.SBPLabels[index,:]
        return img, target
    
    def __len__(self):
        return len(self.Signals)