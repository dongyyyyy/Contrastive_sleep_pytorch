from train.single_epoch.ResNet_dataAugmentation import *
from train.single_epoch.ResNet_contrastive import *
from models.modules.ResNet_module import *
if __name__ == '__main__':
    # methods_list = [['crop'],['cutout'],['permute'],['crop','permute'],['permute','crop'],['cutout','permute'],['permute','cutout'],['cutout','crop'],['crop','cutout'],
    # ['crop','cutout','permute'],['permute','crop','cutout'],['permute','cutout','crop']]
    blocks = Bottleneck
    block_num = [3,4,6,3]
    block_channel=[64,128,128,256]
    first_conv=[49,10,24]
    block_kernel_size = 9
    preprocessing = True
    # for methods in methods_list:
    #     training_ResNet_dataAugmentation(blocks = blocks,block_num =block_num,block_channel = block_channel,first_conv = first_conv,block_kernel_size = block_kernel_size,
    #                                     preprocessing=preprocessing,preprocessing_methods=methods,
    #                                     use_channel=[0,1,2,3,4,5,6,7,8,9],classification_mode='5class',use_dataset='All',gpu_num=[0,1,2,3])
    methods = ['crop','permute']
    
    training_ResNet_contrastiveLearning(blocks = blocks,block_num =block_num,block_channel = block_channel,first_conv = first_conv,block_kernel_size = block_kernel_size,
                                    preprocessing=preprocessing,preprocessing_methods=methods,
                                    use_channel=[0,1,2,3,4,5,6,7,8,9],classification_mode='5class',use_dataset='All',gpu_num=[0,1,2,3])