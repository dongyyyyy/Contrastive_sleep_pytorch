from utils.function.transform import *

if __name__ == '__main__':
    path = '/home/eslab/dataset/seoulDataset/4EEG_2EOG_1EMG_ECG_Flow_Abdomen/signals_dataloader_standard_each_200hz/A2019-NX-01-0001_3_/'
    file_list = os.listdir(path)

    tr = Transform()
    index = 0
    signals = np.load(path+file_list[index])
    
    os.makedirs('/home/eslab/show_transform/',exist_ok=True)
    plt.plot(signals[0,:])
    plt.savefig('/home/eslab/show_transform/origin.png')
    plt.cla()

    plt.plot(tr.cutout_resize(signals,length=1000)[0,:])
    plt.savefig('/home/eslab/show_transform/cutout_resize.png')
    plt.cla()