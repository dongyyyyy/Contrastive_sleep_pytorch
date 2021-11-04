from include.header import *

from utils.function.function import *
from utils.function.dataloader_custom import *
from utils.function.loss_fn import *

from models.cnn.ResNet import *

from utils.function.scheduler import *




def train_ResNet_contrastiveLearning(save_filename,logging_filename, train_dataset_list,val_dataset_list,test_dataset_list,batch_size = 10000,
                                                 epochs=2000,learning_rate=0.001,use_scaling=False,scaling=1e+6,
                                          optim='Adam',loss_function='CE',epsilon=0.7,noise_scale=2e-6,
                                          use_noise=True,preprocessing=False,preprocessing_methods=['crop','permute'],use_channel=[0,1,2],scheduler=None,
                                          warmup_iter=20,cosine_decay_iter=40,stop_iter=300,gamma=0.8,
                                          class_num=5,classification_mode='5class',gpu_num=0,blocks=BasicBlock,block_num=[3,4,6,3],block_channel=[64,128,256,512],first_conv=[49,10,24],
                                          block_kernel_size=9):
    # cpu processor num
    cpu_num = multiprocessing.cpu_count()

    train_dataset = Sleep_Dataset_cnn_withPath_forContrastiveLearning(dataset_list=train_dataset_list,class_num=class_num,
                    use_channel=use_channel,use_cuda = True,preprocessing=preprocessing,
                 epsilon = 0.5,preprocessing_method = preprocessing_methods,permute_size=200,crop_size=4000,cutout_size=1000,classification_mode=classification_mode,loader_mode = 'train')
    
    val_dataset = Sleep_Dataset_cnn_withPath_forContrastiveLearning(dataset_list=val_dataset_list,class_num=class_num,
                    use_channel=use_channel,use_cuda = True,preprocessing=False,
                 epsilon = 0.5,preprocessing_method = preprocessing_methods,permute_size=200,crop_size=4000,cutout_size=1000,classification_mode=classification_mode,loader_mode = 'train')

    weights,count = make_weights_for_balanced_classes(train_dataset.signals_files_path)
    # print(f'weights : {weights} / count : {count}')
    

    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights,len(weights))
    # train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle=True, num_workers=(cpu_num//2))
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size,sampler=sampler,num_workers=(cpu_num//2))

    #dataload Validation Dataset
    

    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=(cpu_num//2))
    
    print(train_dataset.length,val_dataset.length)
    
    # Adam optimizer paramQ
    b1 = 0.5
    b2 = 0.999

    beta = 0.001
    norm_square = 2

    check_file = open(logging_filename, 'w')  # logging file
    temperature  = 0.5
    best_accuracy = 0.
    best_epoch = 0


    model = ResNet_200hz_contrast(block=blocks,layers=block_num, first_conv=first_conv,maxpool=[9,4,4], layer_filters=block_channel, in_channel=len(use_channel),
                    block_kernel_size=block_kernel_size,block_stride_size=2, embedding=512,feature_dim=128,num_classes=class_num, use_batchnorm=True, zero_init_residual=False,
                    groups=1, width_per_group=64, replace_stride_with_dilation=None,
                    norm_layer=None,dropout_p=0.)

                                   
    cuda = torch.cuda.is_available()
    print(f'gpu_num ==> {gpu_num}')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] =f'{gpu_num}'
    print(f'main gpu = {gpu_num[0]}')
    # device = torch.device(f"cuda:{gpu_num[0]}" if torch.cuda.is_available() else "cpu")

    device = torch.device(f"cuda:{gpu_num[0]}" if torch.cuda.is_available() else "cpu")

    # torch.cuda.set_device(device)
    if cuda:
        print('can use CUDA!!!')
        model = model.to(device)

    # exit(1)
    print('torch.cuda.device_count() : ', torch.cuda.device_count())

    if torch.cuda.device_count() > 1:
        print('Multi GPU Activation !!!', torch.cuda.device_count())
        model = nn.DataParallel(model,device_ids=gpu_num)
        # model = nn.DataParallel(model,device_ids=gpu_num)
            # model.to(f'cuda:{model.device_ids[0]}')

    # summary(model, (1, 125*30))
    # model.apply(weights_init)  # weight init
    print('loss function : %s' % loss_function)
    if loss_function == 'CE':
        loss_fn = nn.CrossEntropyLoss().to(device)
    elif loss_function == 'CEW':
        samples_per_cls = count / np.sum(count)
        no_of_classes = class_num
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * no_of_classes
        weights = torch.tensor(weights).float()
        weights = weights.to(device)
        loss_fn = nn.CrossEntropyLoss(weight=weights).to(device)
    elif loss_function == 'FL':
        loss_fn = FocalLoss(gamma=2).to(device)
    elif loss_function == 'CBL':
        samples_per_cls = count / np.sum(count)
        loss_fn = CB_loss(samples_per_cls=samples_per_cls, no_of_classes=class_num, loss_type='focal', beta=0.9999,
                          gamma=2.0)
    elif loss_function == 'Smooth':
        loss_fn = LabelSmoothingCrossEntropy(epsilon=0.1)
    # loss_fn = FocalLoss(gamma=2).to(device)

    # optimizer ADAM (SGD의 경우에는 정상적으로 학습이 진행되지 않았음)
    if optim == 'Adam':
        print('Optimizer : Adam')
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(b1, b2), weight_decay=1e-6)
    elif optim == 'RMS':
        print('Optimizer : RMSprop')
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optim == 'SGD':
        print('Optimizer : SGD')
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5,nesterov=False)
    elif optim == 'AdamW':
        print('Optimizer AdamW')
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(b1, b2))


    gamma = 0.8

    lr = learning_rate
    epochs = epochs
    if scheduler == 'WarmUp_restart_gamma':
        print(f'target lr : {learning_rate} / warmup_iter : {warmup_iter} / cosine_decay_iter : {cosine_decay_iter} / gamma : {gamma}')
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cosine_decay_iter+1)
        scheduler = LearningRateWarmUP_restart_changeMax(optimizer=optimizer,
                                    warmup_iteration=warmup_iter,
                                    cosine_decay_iter=cosine_decay_iter,
                                    target_lr=lr,
                                    after_scheduler=scheduler_cosine,gamma=gamma)
    elif scheduler == 'WarmUp_restart':
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cosine_decay_iter+1)
        scheduler = LearningRateWarmUP_restart(optimizer=optimizer,
                                    warmup_iteration=warmup_iter,
                                    cosine_decay_iter=cosine_decay_iter,
                                    target_lr=lr,
                                    after_scheduler=scheduler_cosine)
    elif scheduler == 'WarmUp':
        print(f'target lr : {learning_rate} / warmup_iter : {warmup_iter}')
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs-warmup_iter+1)
        scheduler = LearningRateWarmUP(optimizer=optimizer,
                                    warmup_iteration=warmup_iter,
                                    target_lr=lr,
                                    after_scheduler=scheduler_cosine)
    elif scheduler == 'StepLR':
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [11, 21], gamma=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif scheduler == 'Reduce':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.5, patience=10,
                                                           min_lr=1e-6)
    elif scheduler == 'Cosine':
        print('Cosine Scheduler')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer, T_max=epochs)


    # scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=.5)
    # loss의 값이 최소가 되도록 하며, 50번 동안 loss의 값이 감소가 되지 않을 경우 factor값 만큼
    # learning_rate의 값을 줄이고, 최저 1e-6까지 줄어들 수 있게 설정
    
    best_loss = 0.
    stop_count = 0
    for epoch in range(epochs):
        scheduler.step(epoch)
        train_total_loss = 0.0

        val_total_loss = 0.0

        start_time = time.time()
        model.train()

        output_str = 'current epoch : %d/%d / current_lr : %f \n' % (epoch+1,epochs,optimizer.state_dict()['param_groups'][0]['lr'])
        sys.stdout.write(output_str)
        check_file.write(output_str)
        index = 0
        with tqdm(train_dataloader,desc='Train',unit='batch') as tepoch:
            for index,(batch_signal1,batch_signal2,_) in enumerate(tepoch):
                batch_signal1 = batch_signal1.to(device)
                batch_signal2 = batch_signal2.to(device)
                
                out_1 = model(batch_signal1)
                out_2 = model(batch_signal2)

                out = torch.cat([out_1,out_2],dim=0)

                
                sim_matrix = torch.exp(torch.mm(out,out.t().contiguous()) / temperature)
                mask = (torch.ones_like(sim_matrix) - torch.eye(2 * len(batch_signal1), device=sim_matrix.device)).bool()

                sim_matrix = sim_matrix.masked_select(mask).view(2 * len(batch_signal1), -1)

                pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)

                # print(f'pos_sim = {pos_sim}')
                
                pos_sim = torch.cat([pos_sim,pos_sim],dim=0)
                loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_total_loss += loss.item()
                
                tepoch.set_postfix(loss=train_total_loss/(index+1))

        train_total_loss /= index
        
        output_str = 'train dataset : %d/%d epochs spend time : %.4f sec / total_loss : %.4f \n' \
                    % (epoch + 1, epochs, time.time() - start_time, train_total_loss)
        # sys.stdout.write(output_str)
        check_file.write(output_str)

        # check validation dataset
        start_time = time.time()
        model.eval()
        index = 0
        with tqdm(val_dataloader,desc='Validation',unit='batch') as tepoch:
            for index,(batch_signal1,batch_signal2,_) in enumerate(tepoch):
                batch_signal1 = batch_signal1.to(device)
                batch_signal2 = batch_signal2.to(device)
                with torch.no_grad():
                    out_1 = model(batch_signal1)
                    out_2 = model(batch_signal2)

                    out = torch.cat([out_1,out_2],dim=0)

                    
                    sim_matrix = torch.exp(torch.mm(out,out.t().contiguous()) / temperature)
                    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * len(batch_signal1), device=sim_matrix.device)).bool()

                    sim_matrix = sim_matrix.masked_select(mask).view(2 * len(batch_signal1), -1)

                    pos_sim = torch.exp(torch.sum(out_1 * out_2,dim=-1) / temperature)

                    pos_sim = torch.cat([pos_sim,pos_sim],dim=0)
                    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

                    val_total_loss += loss.item()

                    tepoch.set_postfix(loss=val_total_loss/(index+1))

        val_total_loss /= index

        output_str = 'val dataset : %d/%d epochs spend time : %.4f sec  / total_loss : %.4f \n' \
                    % (epoch + 1, epochs, time.time() - start_time, val_total_loss)
        # sys.stdout.write(output_str)
        check_file.write(output_str)

        # scheduler.step(float(val_total_loss))
        # scheduler.step(epoch)
        if epoch == 0:
            best_loss = val_total_loss
            best_epoch = epoch
            save_file = save_filename
            # torch.save(model.module.state_dict(), save_file)
            if len(gpu_num) > 1:
                torch.save(model.module.state_dict(), save_file)
            else:
                torch.save(model.state_dict(), save_file)
            stop_count = 0
        else:
            if best_loss > val_total_loss:
                best_loss = val_total_loss
                best_epoch = epoch
                save_file = save_filename
                # torch.save(model.module.state_dict(), save_file)
                if len(gpu_num) > 1:
                    torch.save(model.module.state_dict(), save_file)
                else:
                    torch.save(model.state_dict(), save_file)
                stop_count = 0
            else:
                stop_count += 1
        if stop_count >= stop_iter:
            print('Early Stopping')
            break
        
        output_str = 'best epoch : %d/%d / val loss : %.4f\n' \
                    % (best_epoch + 1, epochs, best_loss)
        sys.stdout.write(output_str)
        print('=' * 30)

    output_str = 'best epoch : %d/%d / loss : %.4f\n' \
                 % (best_epoch + 1, epochs, best_loss)
    sys.stdout.write(output_str)
    check_file.write(output_str)
    print('=' * 30)

    check_file.close()





def training_ResNet_contrastiveLearning(blocks = Bottleneck,block_num = [3,4,6,3],block_channel = [64,128,128,256],first_conv = [49,10,24],block_kernel_size = 9,
                                    preprocessing=False,preprocessing_methods=['crop','permute'],
                                    use_channel=[0,1],classification_mode='5class',use_dataset='SeouUniv',gpu_num=[0]):
    # signals_path = '/home/eslab/dataset/seoulDataset/7channel_prefilter_butter_minmax_-1_1/signals_dataloader/'
    seoul_signals_path = '/home/eslab/dataset/seoulDataset/4EEG_2EOG_1EMG_ECG_Flow_Abdomen/signals_dataloader_standard_each_200hz/'
    hallym_signals_path = '/home/eslab/dataset/hallymDataset/4EEG_2EOG_1EMG_ECG_Flow_Abdomen/signals_dataloader_standard_each_200hz/'
    if use_dataset == 'All' or use_dataset == 'SeoulUniv':
        seoul_dataset_list = os.listdir(seoul_signals_path)
        seoul_dataset_list.sort()
    if use_dataset =='All' or use_dataset == 'HallymUniv':
        hallym_dataset_list = os.listdir(hallym_signals_path)
        hallym_dataset_list.sort()
    
    
    random_seed = 2
    
    random.seed(random_seed) # seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if use_dataset == 'All' or use_dataset == 'SeoulUniv':
        random.shuffle(seoul_dataset_list)
    if use_dataset =='All' or use_dataset == 'HallymUniv':
        random.shuffle(hallym_dataset_list)
    if use_dataset == 'All' or use_dataset == 'SeoulUniv':
        seoul_dataset_list = [seoul_signals_path + filename + '/' for filename in seoul_dataset_list]
        osa_seoul_dataset_list = [[],[],[],[]]
    if use_dataset =='All' or use_dataset == 'HallymUniv':
        hallym_dataset_list = [hallym_signals_path + filename + '/' for filename in hallym_dataset_list]
        osa_hallym_dataset_list = [[],[],[],[]]

    training_fold_list = []
    validation_fold_list = []
    test_fold_list = []
    if use_dataset == 'All' or use_dataset == 'SeoulUniv':
        for dataset in seoul_dataset_list:
            osa_seoul_dataset_list[int(dataset.split('_')[-2])].append(dataset)
    if use_dataset =='All' or use_dataset == 'HallymUniv':
        for dataset in hallym_dataset_list:
            osa_hallym_dataset_list[int(dataset.split('_')[-2])].append(dataset)
    
    seoul_osa_len = []
    hallym_osa_len = []


    if use_dataset == 'All' or use_dataset == 'SeoulUniv':
        for index in range(len(osa_seoul_dataset_list)):
            seoul_osa_len.append(len(osa_seoul_dataset_list[index]))
        print(f'seoul osa dataset list len : {seoul_osa_len}')
    if use_dataset =='All' or use_dataset == 'HallymUniv':
        for index in range(len(osa_hallym_dataset_list)):
            hallym_osa_len.append(len(osa_hallym_dataset_list[index]))
        print(f'hallym osa dataset list len : {hallym_osa_len}')
    

    if use_dataset == 'All' or use_dataset == 'SeoulUniv':
        seoul_train_len = [int(index * 0.8) for index in seoul_osa_len]
        seoul_val_len = [(seoul_osa_len[index] - seoul_train_len[index])//2 for index in range(len(seoul_osa_len))]

    if use_dataset =='All' or use_dataset == 'HallymUniv':
        hallym_train_len = [int(index * 0.8) for index in hallym_osa_len]
        hallym_val_len = [(hallym_osa_len[index] - hallym_train_len[index])//2 for index in range(len(hallym_osa_len))]
    
    
    
    training_fold_list = []
    validation_fold_list = []
    test_fold_list = []
    if use_dataset == 'All' or use_dataset == 'SeoulUniv':
        for osa_index in range(len(osa_seoul_dataset_list)):
            for i in range(0,seoul_train_len[osa_index]):
                training_fold_list.append(osa_seoul_dataset_list[osa_index][i])
            for i in range(seoul_train_len[osa_index],seoul_train_len[osa_index]+seoul_val_len[osa_index]):
                validation_fold_list.append(osa_seoul_dataset_list[osa_index][i])
            for i in range(seoul_train_len[osa_index]+seoul_val_len[osa_index],len(osa_seoul_dataset_list[osa_index])):
                test_fold_list.append(osa_seoul_dataset_list[osa_index][i])    
    if use_dataset =='All' or use_dataset == 'HallymUniv':
        for osa_index in range(len(osa_hallym_dataset_list)):
            for i in range(0,hallym_train_len[osa_index]):
                training_fold_list.append(osa_hallym_dataset_list[osa_index][i])
            for i in range(hallym_train_len[osa_index],hallym_train_len[osa_index]+hallym_val_len[osa_index]):
                validation_fold_list.append(osa_hallym_dataset_list[osa_index][i])
            for i in range(hallym_train_len[osa_index]+hallym_val_len[osa_index],len(osa_hallym_dataset_list[osa_index])):
                test_fold_list.append(osa_hallym_dataset_list[osa_index][i])

    # print(dataset_list[:10])

    print(len(training_fold_list))
    print(len(validation_fold_list))
    print(len(test_fold_list)) 
    
    train_label,train_label_percent = check_label_info_withPath( file_list = training_fold_list)
    val_label,val_label_percent = check_label_info_withPath(file_list = validation_fold_list)
    test_label,test_label_percent = check_label_info_withPath(file_list = test_fold_list)

    
    
    print(train_label)
    print(train_label_percent)
    print(val_label)
    print(val_label_percent)
    print(test_label)
    print(test_label_percent)

    # exit(1)

    # exit(1)
    epochs = 100
    batch_size = 512

    # preprocessing=True
    # preprocessing_methods = ['cutout','permute']


    # learning_rate = 0.0001
    stop_iter = 5
    loss_function = 'CE' # CE
    optim= 'Adam'
    use_noise = False
    epsilon = 0.8
    noise_scale = 2e-6
    scheduler = 'Cosine' # 'WarmUp_restart'
    if classification_mode == '5class':
        class_num = 5
    else:
        class_num = 3
    
    model_save_path = f'/data/hdd3/Contrastive_Sleep/saved_model/11_01/ContrastiveLearning/DataAugmentation/resnet_8_1_1_imageFormat/{blocks}_{block_num}_{block_channel}_{block_kernel_size}/{preprocessing}_{preprocessing_methods}/'
    logging_save_path = f'/data/hdd3/Contrastive_Sleep/log/11_01/ContrastiveLearning/DataAugmentation/resnet_8_1_1_imageFormat/{blocks}_{block_num}_{block_channel}_{block_kernel_size}/{preprocessing}_{preprocessing_methods}/'
    os.makedirs(model_save_path,exist_ok=True)
    os.makedirs(logging_save_path,exist_ok=True)
    if optim == 'SGD':
        learning_rate = 0.1
    if optim == 'Adam':
        learning_rate = 0.0001
    save_filename = model_save_path + f'resnet_SimCLR_5classes_Adam_{use_channel}_{use_dataset}_{first_conv}.pth'
    logging_filename = logging_save_path + f'resnet_SimCLR_5classes_Adam_{use_channel}_{use_dataset}_{first_conv}.txt'


    print('save filename : ',save_filename)
    print(save_filename)
    
    train_ResNet_contrastiveLearning(save_filename=save_filename,logging_filename=logging_filename, train_dataset_list=training_fold_list,val_dataset_list=validation_fold_list,
                                                        test_dataset_list=test_fold_list,epochs=epochs,batch_size=batch_size,learning_rate=learning_rate,
                                                        optim=optim,loss_function=loss_function,epsilon=epsilon,noise_scale=noise_scale,class_num = class_num,
                                                        use_noise=use_noise,preprocessing=preprocessing,preprocessing_methods=preprocessing_methods,scheduler=scheduler,stop_iter=stop_iter,use_channel=use_channel,
                                                        classification_mode=classification_mode,gpu_num=gpu_num,blocks=blocks,block_num=block_num,block_channel=block_channel,
                                                        first_conv=first_conv,block_kernel_size=block_kernel_size)

