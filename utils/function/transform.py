import numpy as np
from utils.function.function import *

class Transform:
    def __init__(self):
        pass
    def add_jittering(self, signal, mu=0,std=1,channel_dependent=False):
        if channel_dependent == False: # Channel Independent Noise
            noise = np.random.normal(loc=mu,scale=std,size=np.shape(signal))
        else: # Channel Dependent Noise
            noise = np.random.normal(loc=mu,scale=std,size=np.shape(signal)[-1])
            noise = np.reshape((-1,1))
            noise = np.repeat(noise,repeats=np.shape(signal)[0],axis=1)
            noise = noise.T
        signal = signal + noise
        return signal

    def horizon_flip(self,signal):
        signal = np.flip(signal,axis=-1)
        return signal
    
    def permute(self,signal,pieces_size=200):
        assert np.shape(signal)[-1] % pieces_size == 0, "Fault Pieces Size!!!"
        permute_length = np.shape(signal)[-1]//pieces_size
        random_index = np.arange(permute_length)
        
        np.random.shuffle(random_index)
        permute_signal = signal.reshape(np.shape(signal)[0],permute_length,pieces_size)

        permute_signal = permute_signal[:,random_index,:]

        permute_signal = permute_signal.reshape(np.shape(signal)[0],-1)

        return permute_signal

    def cutout_resize(self, signal, length):
        cutout_length = int(length)
        while(1):
            random_num = np.random.rand()
            # print(f'np.random.rand() = {np.random.rand()}')
            # print(np.shape(signal)[-1])
            # print(f'random_num = {random_num} // int(np.ceil(np.shape(signal)[-1] // 100 * np.random.rand())) = {int(np.ceil(np.shape(signal)[-1] // 100 * random_num))}')
            start_num = int(np.ceil(np.shape(signal)[-1] * random_num))
            if start_num + cutout_length <= np.shape(signal)[-1]:
                break
        # print(f'start_num = {start_num}')
        if start_num + cutout_length == np.shape(signal)[-1]:
            cutout_signal = signal[:,:start_num]
        else:
            cutout_signal = np.concatenate((signal[:,:start_num],signal[:,start_num+cutout_length:]),axis=1)
        
        cutout_signal = interp_1d_multiChannel(arr=cutout_signal,short_sample=6000-length,long_sample=6000)

        return cutout_signal
    
    def crop_resize(self, signal, length):
        crop_length = int(length)
        while(1):
            start_num = int(np.ceil(np.shape(signal)[-1] * np.random.rand()))
            if start_num + crop_length <= np.shape(signal)[-1]:
                break
        crop_signal = signal[:,start_num:start_num+crop_length]

        crop_signal = interp_1d_multiChannel(arr=crop_signal,short_sample=crop_length,long_sample=6000)

        return crop_signal