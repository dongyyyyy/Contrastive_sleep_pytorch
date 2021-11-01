from train.single_epoch.ResNet_dataAugmentation import *

if __name__ == '__main__':
    training_ResNet_dataAugmentation(use_channel=[0,1,2,3,4,5,6,7,8,9],classification_mode='5class',use_dataset='All',gpu_num=[0])