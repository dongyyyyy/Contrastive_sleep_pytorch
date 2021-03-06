from utils.function.function import *
from utils.function.transform import *

def make_weights_for_balanced_classes(data_list, nclasses=5,check_file='.npy'):
    count = [0] * nclasses
    
    for data in data_list:
        count[int(data.split(check_file)[0].split('_')[-1])] += 1

    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(data_list)
    for idx, val in enumerate(data_list):
        weight[idx] = weight_per_class[int(val.split(check_file)[0].split('_')[-1])]
    return weight , count

class Sleep_Dataset_cnn_withPath(object):
    def read_dataset(self):
        all_signals_files = []
        all_labels = []

        for dataset_folder in self.dataset_list:
            signals_path = dataset_folder
            signals_list = os.listdir(signals_path)
            signals_list.sort()
            for signals_filename in signals_list:
                signals_file = signals_path+signals_filename
                all_signals_files.append(signals_file)
                all_labels.append(int(signals_filename.split('.npy')[0].split('_')[-1]))
                
        return all_signals_files, all_labels, len(all_signals_files)

    def __init__(self,
                 dataset_list,
                 class_num=5,
                 use_channel=[0,1,2],
                 use_cuda = True,
                 sample_rate=200,
                 epoch_size=30,
                 preprocessing=False,
                 epsilon = 0.5,
                 preprocessing_method = ['permute','crop'],
                 permute_size=200,
                 crop_size=1000,
                 cutout_size=1000,
                 classification_mode='5class',
                 loader_mode = 'train'
                 ):
        self.class_num = class_num
        self.dataset_list = dataset_list
        self.signals_files_path, self.labels, self.length = self.read_dataset()


        self.use_channel = use_channel
        self.use_cuda = use_cuda

        self.preprocessing = preprocessing
        self.epsilon = epsilon
        self.preprocessing_method = preprocessing_method
        self.Transform = Transform()
        self.permute_size = permute_size
        self.crop_size = crop_size
        self.cutout_size = cutout_size
        self.loader_mode = loader_mode
        self.classification_mode = classification_mode
        print('classification_mode : ',classification_mode)


    def __getitem__(self, index):

        # current file index

        labels = int(self.labels[index])

        if self.classification_mode == 'REM-NoneREM':
            if labels == 0: # Wake
                labels = 0
            elif labels == 4: #REM
                labels = 2
            else: # None-REM
                labels = 1

        elif self.classification_mode == 'LS-DS':
            if labels == 0:
                labels = 0
            elif labels == 1 or labels == 2:
                labels = 1
            else:
                labels = 2
        signals = None
        count = 0
        
        signals = np.load(self.signals_files_path[index])
        signals = signals[self.use_channel,:]
        

        # for i in range(self.seq_size):
        #     print(np.array_equal(signals[:,i*self.stride:(i*self.stride)+self.window_size],input_signals[i]))              

        signals = np.array(signals)
        if self.loader_mode:
            if self.preprocessing:
                for current_method in self.preprocessing_method:
                    if current_method == 'permute':
                        if np.random.rand() < self.epsilon:
                            signals = self.Transform.permute(signal=signals,pieces_size=self.permute_size)
                    elif current_method == 'crop':
                        if np.random.rand() < self.epsilon:
                            signals = self.Transform.crop_resize(signal=signals,length=self.crop_size)
                    elif current_method =='cutout':
                        if np.random.rand() < self.epsilon:
                            signals = self.Transform.cutout_resize(signal=signals,length=self.cutout_size)

        if self.use_cuda:
            signals = torch.from_numpy(signals).float()
        
        return signals,labels
        
    def __len__(self):
        return self.length 

class Sleep_Dataset_cnn_withPath_withoutAugmentation(object):
    def read_dataset(self):
        all_signals_files = []
        all_labels = []

        for dataset_folder in self.dataset_list:
            signals_path = dataset_folder
            signals_list = os.listdir(signals_path)
            signals_list.sort()
            for signals_filename in signals_list:
                signals_file = signals_path+signals_filename
                all_signals_files.append(signals_file)
                all_labels.append(int(signals_filename.split('.npy')[0].split('_')[-1]))
                
        return all_signals_files, all_labels, len(all_signals_files)

    def __init__(self,
                 dataset_list,
                 class_num=5,
                 use_channel=[0,1,2],
                 use_cuda = True,
                 sample_rate=200,
                 epoch_size=30,
                 classification_mode='5class',
                 ):
        self.class_num = class_num
        self.dataset_list = dataset_list
        self.signals_files_path, self.labels, self.length = self.read_dataset()


        self.use_channel = use_channel
        self.use_cuda = use_cuda

        self.classification_mode = classification_mode
        print('classification_mode : ',classification_mode)


    def __getitem__(self, index):

        # current file index

        labels = int(self.labels[index])

        if self.classification_mode == 'REM-NoneREM':
            if labels == 0: # Wake
                labels = 0
            elif labels == 4: #REM
                labels = 2
            else: # None-REM
                labels = 1

        elif self.classification_mode == 'LS-DS':
            if labels == 0:
                labels = 0
            elif labels == 1 or labels == 2:
                labels = 1
            else:
                labels = 2
        signals = None
        count = 0
        
        signals = np.load(self.signals_files_path[index])
        signals = signals[self.use_channel,:]
        

        # for i in range(self.seq_size):
        #     print(np.array_equal(signals[:,i*self.stride:(i*self.stride)+self.window_size],input_signals[i]))              

        signals = np.array(signals)
        
        if self.use_cuda:
            signals = torch.from_numpy(signals).float()
        
        return signals,labels
        
    def __len__(self):
        return self.length 

class Sleep_Dataset_cnn_window_withPath(object):
    def read_dataset(self):
        all_signals_files = []
        all_labels = []

        for dataset_folder in self.dataset_list:
            signals_path = dataset_folder
            signals_list = os.listdir(signals_path)
            signals_list.sort()
            for signals_filename in signals_list:
                signals_file = signals_path+signals_filename
                all_signals_files.append(signals_file)
                all_labels.append(int(signals_filename.split('.npy')[0].split('_')[-1]))
                
        return all_signals_files, all_labels, len(all_signals_files)

    def __init__(self,
                 dataset_list,
                 class_num=5,
                 use_channel=[0,1],
                 use_cuda = True,
                 window_size=400,
                 stride=200,
                 sample_rate=200,
                 epoch_size=30,
                 classification_mode='5class'
                 ):
        self.class_num = class_num
        self.dataset_list = dataset_list
        self.signals_files_path, self.labels, self.length = self.read_dataset()


        self.use_channel = use_channel
        self.use_cuda = use_cuda
        self.seq_size = ((sample_rate*epoch_size)-window_size)//stride + 1
        self.window_size = window_size
        self.stride = stride

        self.classification_mode = classification_mode
        print('classification_mode : ',classification_mode)
        print(f'window size = {window_size} / stride = {stride}')

    def __getitem__(self, index):

        # current file index

        labels = int(self.labels[index])

        if self.classification_mode == 'REM-NoneREM':
            if labels == 0: # Wake
                labels = 0
            elif labels == 4: #REM
                labels = 2
            else: # None-REM
                labels = 1

        elif self.classification_mode == 'LS-DS':
            if labels == 0:
                labels = 0
            elif labels == 1 or labels == 2:
                labels = 1
            else:
                labels = 2
        signals = None
        count = 0
        

        signals = []
        c_signals = np.load(self.signals_files_path[index])
        # print('1 : ',c_signals.shape)
        c_signals = c_signals[self.use_channel,:]
        # print('2 : ',c_signals.shape)
        for inner_i in range(self.seq_size):
            temp = c_signals[:,inner_i*self.stride:(inner_i*self.stride)+self.window_size]
            signals.append(temp)
            # print(temp.shape)
        # for i in range(self.seq_size):
        #     print(np.array_equal(signals[:,i*self.stride:(i*self.stride)+self.window_size],input_signals[i]))              

        signals = np.array(signals)

        if self.use_cuda:
            signals = torch.from_numpy(signals).float()

            
        count += 1
        
        return signals,labels
        
    def __len__(self):
        return self.length 

class Sleep_Dataset_cnn_sequence_withPath(object):
    def read_dataset(self):
        all_signals_files = []
        all_labels = []

        for dataset_folder in self.dataset_list:
            signals_path = dataset_folder
            signals_list = os.listdir(signals_path)
            signals_list.sort()
            for signals_filename in signals_list:
                signals_file = signals_path+signals_filename
                all_signals_files.append(signals_file)
                all_labels.append(int(signals_filename.split('.npy')[0].split('_')[-1]))
                
        return all_signals_files, all_labels, len(all_signals_files)

    def __init__(self,
                 dataset_list,
                 class_num=5,
                 use_channel=[0,1],
                 use_cuda = True,
                 sequence_length=3,
                 start_epoch = -1,
                 end_epoch = 2,
                 classification_mode='5class'
                 ):
        self.sequence_length=sequence_length
        self.class_num = class_num
        self.dataset_list = dataset_list
        self.signals_files_path, self.labels, self.length = self.read_dataset()

      

        self.use_channel = use_channel
        self.use_cuda = use_cuda
        
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.classification_mode = classification_mode
        print('classification_mode : ',classification_mode)
        
    def __getitem__(self, index):
        folder_name = '/'.join(self.signals_files_path[index].split('/')[:-1]) + '/'
        folder_list = os.listdir(folder_name)
        folder_list.sort()
        folder_length = len(folder_list)-1
        # current file index

        current_epoch = int(self.signals_files_path[index].split('/')[-1].split('_')[0])
        
        start_epoch = current_epoch + self.start_epoch # ??????
        end_epoch = current_epoch + self.end_epoch # ??????
        labels = int(folder_list[current_epoch].split('.npy')[0].split('_')[-1])

        if self.classification_mode == 'REM-NoneREM':
            if labels == 0: # Wake
                labels = 0
            elif labels == 4: #REM
                labels = 2
            else: # None-REM
                labels = 1
        elif self.classification_mode == 'LS-DS':
            if labels == 0:
                labels = 0
            elif labels == 1 or label == 2:
                labels = 1
            else:
                labels = 2
        signals = None
        count = 0
        
        for i in range(start_epoch,end_epoch,1):
            if i <= 0:
                c_signals = np.load(folder_name+folder_list[0])
                c_signals = c_signals[self.use_channel,:]             
                input_signals = np.array(c_signals)
                
            elif i >= folder_length:
                c_signals = np.load(folder_name+folder_list[folder_length])
                c_signals = c_signals[self.use_channel,:]
                input_signals = np.array(c_signals)
            else:
                c_signals = np.load(folder_name+folder_list[i])
                c_signals = c_signals[self.use_channel,:]
                input_signals = np.array(c_signals)

            if self.use_cuda:
                    input_signals = torch.from_numpy(input_signals).float()

            if count == 0:
                signals = input_signals.unsqueeze(0)
            else:
                signals = torch.cat((signals,input_signals.unsqueeze(0)))
             
            count += 1
        
        return signals,labels
        
    def __len__(self):
        return self.length 

class Sleep_Dataset_stft_sequence_withPath(object):
    def read_dataset(self):
        all_signals_files = []
        all_labels = []

        for dataset_folder in self.dataset_list:
            signals_path = dataset_folder
            signals_list = os.listdir(signals_path)
            signals_list.sort()
            for signals_filename in signals_list:
                signals_file = signals_path+signals_filename
                all_signals_files.append(signals_file)
                all_labels.append(int(signals_filename.split('.npy')[0].split('_')[-1]))
                
        return all_signals_files, all_labels, len(all_signals_files)

    def __init__(self,
                 dataset_list,
                 class_num=5,
                 use_channel=[0,1],
                 use_cuda = True,
                 sequence_length=3,
                 start_epoch = -1,
                 end_epoch = 2,
                 classification_mode='5class'
                 ):
        self.sequence_length=sequence_length
        self.class_num = class_num
        self.dataset_list = dataset_list
        self.signals_files_path, self.labels, self.length = self.read_dataset()

      

        self.use_channel = use_channel
        self.use_cuda = use_cuda
        
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.classification_mode = classification_mode
        print('classification_mode : ',classification_mode)
        
    def __getitem__(self, index):
        folder_name = '/'.join(self.signals_files_path[index].split('/')[:-1]) + '/'
        folder_list = os.listdir(folder_name)
        folder_list.sort()
        folder_length = len(folder_list)-1
        # current file index

        current_epoch = int(self.signals_files_path[index].split('/')[-1].split('_')[0])
        
        start_epoch = current_epoch + self.start_epoch # ??????
        end_epoch = current_epoch + self.end_epoch # ??????
        labels = int(folder_list[current_epoch].split('.npy')[0].split('_')[-1])

        if self.classification_mode == 'REM-NoneREM':
            if labels == 0: # Wake
                labels = 0
            elif labels == 4: #REM
                labels = 2
            else: # None-REM
                labels = 1
        elif self.classification_mode == 'LS-DS':
            if labels == 0:
                labels = 0
            elif labels == 1 or label == 2:
                labels = 1
            else:
                labels = 2
        signals = None
        count = 0
        
        for i in range(start_epoch,end_epoch,1):
            if i <= 0:
                c_signals = np.load(folder_name+folder_list[0])                       
                input_signals = np.array(c_signals)
                
            elif i >= folder_length:
                c_signals = np.load(folder_name+folder_list[folder_length])
                input_signals = np.array(c_signals)
            else:
                c_signals = np.load(folder_name+folder_list[i])
                input_signals = np.array(c_signals)

            if self.use_cuda:
                    input_signals = torch.from_numpy(input_signals).float()

            if count == 0:
                signals = input_signals.unsqueeze(0)
            else:
                signals = torch.cat((signals,input_signals.unsqueeze(0)))
             
            count += 1
    
        return signals,labels
        
    def __len__(self):
        return self.length 


class Sleep_Dataset_cnn_window_sequence_withPath(object):
    def read_dataset(self):
        all_signals_files = []
        all_labels = []

        for dataset_folder in self.dataset_list:
            signals_path = dataset_folder
            signals_list = os.listdir(signals_path)
            signals_list.sort()
            for signals_filename in signals_list:
                signals_file = signals_path+signals_filename
                all_signals_files.append(signals_file)
                all_labels.append(int(signals_filename.split('.npy')[0].split('_')[-1]))
                
        return all_signals_files, all_labels, len(all_signals_files)

    def __init__(self,
                 dataset_list,
                 class_num=5,
                 use_channel=[0,1,2],
                 use_cuda = True,
                 window_size=400,
                 stride=200,
                 sample_rate=200,
                 epoch_size=30,
                 sequence_length=3,
                 start_epoch = -1,
                 end_epoch = 2,
                 classification_mode='5class'
                 ):
        self.sequence_length=sequence_length
        self.class_num = class_num
        self.dataset_list = dataset_list
        self.signals_files_path, self.labels, self.length = self.read_dataset()        

        self.use_channel = use_channel
        self.use_cuda = use_cuda
        self.seq_size = ((sample_rate*epoch_size)-window_size)//stride + 1
        self.window_size = window_size
        self.stride = stride
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.classification_mode = classification_mode
        print('classification_mode : ',classification_mode)
        print(f'window size = {window_size} / stride = {stride}')
        print('start epoch : ', self.start_epoch)
        print('end epoch : ',self.end_epoch)
    def __getitem__(self, index):
        folder_name = '/'.join(self.signals_files_path[index].split('/')[:-1]) + '/'
        folder_list = os.listdir(folder_name)
        folder_list.sort()
        folder_length = len(folder_list)-1
        # current file index

        current_epoch = int(self.signals_files_path[index].split('/')[-1].split('_')[0])
        
        start_epoch = current_epoch + self.start_epoch # ??????
        end_epoch = current_epoch + self.end_epoch # ??????
        labels = int(folder_list[current_epoch].split('.npy')[0].split('_')[-1])

        if self.classification_mode == 'REM-NoneREM':
            if labels == 0: # Wake
                labels = 0
            elif labels == 4: #REM
                labels = 2
            else: # None-REM
                labels = 1
        elif self.classification_mode == 'LS-DS':
            if labels == 0:
                labels = 0
            elif labels == 1 or labels == 2:
                labels = 1
            else:
                labels = 2
        signals = None
        count = 0
        
        for i in range(start_epoch,end_epoch,1):
            if i <= 0:
                input_signals = []
                c_signals = np.load(folder_name+folder_list[0])
                c_signals = c_signals[self.use_channel,:]
                for inner_i in range(self.seq_size):
                    temp = c_signals[:,inner_i*self.stride:(inner_i*self.stride)+self.window_size]
                    input_signals.append(temp)

                # for i in range(self.seq_size):
                #     print(np.array_equal(signals[:,i*self.stride:(i*self.stride)+self.window_size],input_signals[i]))              

                input_signals = np.array(input_signals)
                
            elif i >= folder_length:
                input_signals = []
                c_signals = np.load(folder_name+folder_list[folder_length])
                c_signals = c_signals[self.use_channel,:]
                for inner_i in range(self.seq_size):
                    temp = c_signals[:,inner_i*self.stride:(inner_i*self.stride)+self.window_size]
                    input_signals.append(temp)

                # for i in range(self.seq_size):
                #     print(np.array_equal(signals[:,i*self.stride:(i*self.stride)+self.window_size],input_signals[i]))              

                input_signals = np.array(input_signals)
                # print('input signals shape : ',input_signals.shape)
            else:
                input_signals = []
                c_signals = np.load(folder_name+folder_list[i])
                c_signals = c_signals[self.use_channel,:]
                for inner_i in range(self.seq_size):
                    temp = c_signals[:,inner_i*self.stride:(inner_i*self.stride)+self.window_size]
                    input_signals.append(temp)

                # for i in range(self.seq_size):
                #     print(np.array_equal(signals[:,i*self.stride:(i*self.stride)+self.window_size],input_signals[i]))              

                input_signals = np.array(input_signals)

            if self.use_cuda:
                    input_signals = torch.from_numpy(input_signals).float()

            if count == 0:
                signals = input_signals.unsqueeze(0)
            else:
                signals = torch.cat((signals,input_signals.unsqueeze(0)))
             
            count += 1
        
        return signals,labels
        
    def __len__(self):
        return self.length 

class Sleep_Dataset_cnn_withPath_forContrastiveLearning(object):
    def read_dataset(self):
        all_signals_files = []
        all_labels = []

        for dataset_folder in self.dataset_list:
            signals_path = dataset_folder
            signals_list = os.listdir(signals_path)
            signals_list.sort()
            for signals_filename in signals_list:
                signals_file = signals_path+signals_filename
                all_signals_files.append(signals_file)
                all_labels.append(int(signals_filename.split('.npy')[0].split('_')[-1]))
                
        return all_signals_files, all_labels, len(all_signals_files)

    def __init__(self,
                 dataset_list,
                 class_num=5,
                 use_channel=[0,1,2],
                 use_cuda = True,
                 sample_rate=200,
                 epoch_size=30,
                 preprocessing=False,
                 epsilon = 0.5,
                 preprocessing_method = ['permute','crop'],
                 permute_size=200,
                 crop_size=1000,
                 cutout_size=1000,
                 classification_mode='5class',
                 loader_mode = 'train'
                 ):
        self.class_num = class_num
        self.dataset_list = dataset_list
        self.signals_files_path, self.labels, self.length = self.read_dataset()


        self.use_channel = use_channel
        self.use_cuda = use_cuda

        self.preprocessing = preprocessing
        self.epsilon = epsilon
        self.preprocessing_method = preprocessing_method
        self.Transform = Transform()
        self.permute_size = permute_size
        self.crop_size = crop_size
        self.cutout_size = cutout_size
        self.loader_mode = loader_mode
        self.classification_mode = classification_mode
        print('classification_mode : ',classification_mode)


    def __getitem__(self, index):

        # current file index

        labels = int(self.labels[index])

        if self.classification_mode == 'REM-NoneREM':
            if labels == 0: # Wake
                labels = 0
            elif labels == 4: #REM
                labels = 2
            else: # None-REM
                labels = 1

        elif self.classification_mode == 'LS-DS':
            if labels == 0:
                labels = 0
            elif labels == 1 or labels == 2:
                labels = 1
            else:
                labels = 2
        signals = None
        count = 0
        
        signals = np.load(self.signals_files_path[index])
        signals = signals[self.use_channel,:]
        

        # for i in range(self.seq_size):
        #     print(np.array_equal(signals[:,i*self.stride:(i*self.stride)+self.window_size],input_signals[i]))              

        signals = np.array(signals)
        if self.loader_mode:
            if self.preprocessing:
                for signal_index, current_method in enumerate(self.preprocessing_method):
                    if current_method == 'permute':
                        if signal_index == 0:
                            signals1 = self.Transform.permute(signal=signals,pieces_size=self.permute_size)
                        else:
                            signals2 = self.Transform.permute(signal=signals,pieces_size=self.permute_size)
                    elif current_method == 'crop':
                        if signal_index == 0:
                            signals1 = self.Transform.crop_resize(signal=signals,length=self.crop_size)
                        else:
                            signals1 = self.Transform.crop_resize(signal=signals,length=self.crop_size)
                    elif current_method =='cutout':
                        if signal_index == 0:
                            signals1 = self.Transform.cutout_resize(signal=signals,length=self.cutout_size)
                        else:
                            signals2 = self.Transform.cutout_resize(signal=signals,length=self.cutout_size)

        if self.use_cuda:
            signals1 = torch.from_numpy(signals1).float()
            signals2 = torch.from_numpy(signals2).float()
        
        return signals1, signals2, labels
        
    def __len__(self):
        return self.length 