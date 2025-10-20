import numpy as np
import torch.nn as nn
import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
import torchvision.transforms as transforms
import argparse
import sys
sys.path.insert(0, '/mnt/home/zkadkhodaie/projects/shared_code')
from network import *
from model_loader_func import *
from trainer import run_training
from dataloader_func import weights_init_kaiming 
from  synthetic_data_generators import oval_dataset, generate_circles_texture, generate_squares_texture, generate_ovals_texture, generate_object_texture_mix
import pickle

####################################################################################################################
########################################################## Experiment specific functions #######################
####################################################################################################################
def repeat_images(train_set,args, N_total): 
    '''
    train_set: a list of tensors. Each tensor a separate class
    N_total: total number of images in the final training set 
    '''
    train_set_repeated = []
    
    n = int(N_total/args.set_size)
    for set in train_set: 
        set = torch.tile(set,(n,1,1,1)  )
        train_set_repeated.append(set)
    return train_set_repeated

####### data prep functions for different datasets #########



def bimodal_Gaussian_dataset( N): 
    '''
    im: mean of the gaussian 
    sigma: variance of the gaussian
    N:number of images in the dataset 
    '''
    im1 = torch.load('../window.pt', weights_only=True)
    im2 = torch.load('../window_rotated.pt', weights_only=True)
    
    diff = (im2 - im1)
    diff = diff/diff.norm()

    sigma = 2
    
    
    data1 = im1 + diff * sigma * torch.randn(N,1,1,1).expand(N,im1.shape[1],im1.shape[2] , im1.shape[3])
    data2 = im2 + diff * sigma * torch.randn(N,1,1,1).expand(N,im2.shape[1],im2.shape[2] , im2.shape[3])
    return [data1, data2]
    


def load_prep_half_bed_half_face(args): 
    '''
    '''
    train_set = []
    test_set = []
    for category in ['img_align_celeba','bedroom']:
        data = torch.load( args.data_root_path  + '/'+ category +'/train_color_80x80.pt', weights_only=True)    
        
        if args.debug: 
            N = args.batch_size
        else: 
            N = 200_000
            
        if args.swap is True: 
            train = data[-N::]    
            test = data[0:100]   
         
        else: 
            train = data[0:N]    
            test = data[-100::]

        
        ### append 
        train_set.append(train)
        test_set.append(test)
        
    
    return train_set, test_set
    

def load_prep_multi_class_data(args): 
    '''
    '''
    train_set = []
    test_set = []
    
    if args.debug: 
        print('debug mode')            
        N = 100
        data = torch.load( args.data_root_path  + '/'+ 'afhq' +'/train_color_80x80.pt', weights_only=True)    
        train = data[0:N]    
        test = data[-100::]
        ### append 
        train_set.append(train)
        test_set.append(test)        
    else:    
        for category in ['img_align_celeba','bedroom', 'afhq', 'flowers102', 'stanford_car', 'nabirds']:
            data = torch.load( args.data_root_path  + '/'+ category +'/train_color_80x80.pt', weights_only=True)    
            
            if category == 'img_align_celeba' or category == 'bedroom': 
                N = 100_000
            elif  category == 'nabirds':
                N = 30000
            elif category == 'afhq' or category == 'stanford_car' : 
                N = 16000
            elif category == 'flowers102' : 
                N = 8000

            if args.swap is True: 
                train = data[-N::]    
                test = data[0:100]            
            else: 
                 train = data[0:N]    
                 test = data[-100::]
            
            ### append 
            train_set.append(train)
            test_set.append(test)
        
    return train_set, test_set
    

    
def load_prep_specific_class(args): 
    '''
    '''
    args.data_name = ['img_align_celeba', 'bedroom', 'nabirds', 'zk-wood-textures'][args.SLURM_ARRAY_TASK_ID]
    args.data_path = args.data_root_path + args.data_name
    
    train_set = []
    test_set = []

    data = torch.load( args.data_path +'/train_80x80.pt', weights_only=True)    
    
    if args.debug: 
        N = args.batch_size
    else: 
        # if data.shape[0] > 120000: 
        #     N = 130000
        # else: 
        N = data.shape[0] - 1000  

    train_set = [data[0:N]]    
    test_set = [data[N::]]
        
    return train_set, test_set


def load_imagenet_subset(args, n_classes,grayscale=False): 

    
    args.data_path = args.data_root_path + args.data_name
    train_set = torch.load( args.data_path + '/train_color_full_64x64_list.pt', weights_only=True) 
    test_set = torch.load( args.data_path + '/validation_color_full_64x64_list.pt', weights_only=True)    
    if len(train_set) != n_classes:
        ids = torch.randint(low=0,high=len(train_set),size = (n_classes,)) # change this to remove repeated random integers 
        args.imagenet_subset_ids = ids 
        if grayscale is False:
            train_set = [train_set[i] for i in ids]
            test_set = [test_set[i] for i in ids]
        else: 
            train_set = [train_set[i].mean(dim=1, keepdim=True) for i in ids]
            test_set = [test_set[i].mean(dim=1, keepdim=True) for i in ids]
            
    return train_set, test_set
    
def load_prep_texture(args, color, dim=80, n_test=4): 
    '''
    returns lists of tensors. Each tensor in the list contains all patches from the same image. 
    '''
    # ### for 80x80 patches
    args.data_path = args.data_root_path + 'texture_EPS'
    if dim==80:
        if color: 
            data = torch.load( args.data_path + '/patched_1024x1024_to_80x80_all_sets_color.pt', weights_only=True)
            n_patches = int(1024/dim)**2 # 144
            
        else:     
            data1 = torch.load( args.data_path + '/patched_1024x1024_to_80x80.pt', weights_only=True)
            data2 = torch.load( args.data_path + '/patched_1024x1024_to_80x80_down.pt', weights_only=True)    
            data = torch.cat([data1, data2])
            n_patches = int(1024/dim)**2 # 144
            
    ## for 128x128 patches             
    if dim==128: 
        if color: 
            data = torch.load( args.data_path + '/patched_1024x1024_to_128x128_all_sets_color.pt', weights_only=True)
            n_patches = int(1024/dim)**2 # 64
            
        else: 
            data = torch.load( args.data_path + '/patched_1536x1536_to_128x128.pt', weights_only=True)
            n_patches = int(1536/dim)**2 # 144
            
    num_classes = int(data.shape[0]/n_patches) # for 80x80: 882 + 786 # for 128x128:786
    
    all_data_train = []
    all_data_test = []
    
    if args.debug: 
        num_classes = 5
    for d in range(num_classes): 
        all_data_train.append(data[n_patches * d: (n_patches * (d+1)) -n_test ])
        all_data_test.append(data[(n_patches * (d+1)) -n_test : (n_patches * (d+1)) ] ) # in each image, leave the last 4 patches for test set       

    if color is False:
        # add my wood texture images     
        data_zk = torch.load( args.data_root_path + '/zk-wood-textures/train_80x80.pt', weights_only=True)
        all_data_train.append(data_zk[0:n_patches-n_test]) #the first 140 patches of  the entire dataset 
        all_data_test.append(data_zk[-n_test::]) #the last 4 of the entire dataset 
    
    return all_data_train, all_data_test





####################################################################################################################
################################################# main #################################################
####################################################################################################################

def main():

    path = '/mnt/home/zkadkhodaie/ceph/22_representation_in_UNet_denoiser/denoisers//UNet_flex/texture_EPS_color/1to765_RF_84x84_set_size_237580_color_no_skip_deep_dec_80x80/'
    
    
    
    ######### load a pretrained model #########
    model, args = load_learned_model(path, print_args=False, new_arch=None, return_args=True)
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    if torch.cuda.is_available():
        print('[ Using CUDA ]')
        model = nn.DataParallel(model).cuda()
    print('number of parameters is ' , sum(p.numel() for p in model.parameters() if p.requires_grad))

    
    ######### modify arguments used for fine tuning #########
    args.debug = False
    args.lr  = args.lr / (2**((args.num_epochs/args.lr_freq ) -1 ))
    # args.lr  = args.lr / (2**2 )    
    args.batch_size = 4096
    args.num_epochs = 1000
    args.lr_freq = 3000
    
    # args.noise_level_range = [1,255]
    args.sigma_dist = 'inv_sqrt'
    
    args.dir_name = args.dir_name + '_more_epochs'
    if not os.path.exists(args.dir_name):
        os.makedirs(args.dir_name)
        
    ## save model args in case 
    with open( args.dir_name +'/exp_arguments.pkl', 'wb') as f:
        pickle.dump(args.__dict__, f)
    
    ######### load raw data #########

    if args.debug: 
        args.num_epochs=3

    
    if 'texture' in args.data_name.split('_'):
        train_set, test_set = load_prep_texture(args, color=True,dim=80, n_test=4)
    
    elif args.data_name == 'imagenet':
        args.data_path = args.data_root_path + args.data_name

        if args.debug: 
            print('debug mode')
            train_set = torch.load( args.data_path + '/validation_color_full_64x64_list.pt', weights_only=True) [0:3]
            test_set = torch.load( args.data_path + '/validation_color_full_64x64_list.pt', weights_only=True) [3:6]
        else: 
            train_set = torch.load( args.data_path + '/train_color_full_64x64_list.pt', weights_only=True) ### 
            test_set = torch.load( args.data_path + '/validation_color_full_64x64_list.pt', weights_only=True)    
            # train_set, test_set = load_imagenet_subset(args, 200)
    
    elif args.data_name == 'face_bedroom': 
        train_set, test_set = load_prep_half_bed_half_face(args)

    elif args.data_name == 'multi_class_dataset': 
        train_set, test_set = load_prep_multi_class_data(args)

    elif args.data_name == 'bedroom': 
        data = torch.load( args.data_root_path  + '/bedroom/train_color_80x80.pt', weights_only=True)    
        train_set = [data[0:150_000]]
        test_set = [data[0:100]]

    elif args.data_name == 'img_align_celeba': 
        data = torch.load( args.data_root_path  + '/img_align_celeba/train_color_80x80.pt', weights_only=True)    
        train_set = [data[0:150_000]]
        test_set = [data[0:100]]
        
    image_size = torch.cat(train_set).shape 
    print('train data size: ', image_size )
    
    ######### select criterion and optimizer #########
    criterion = nn.MSELoss(reduction='sum')
    optimizer = Adam(filter(lambda p: p.requires_grad,model.parameters()), lr = args.lr)



   
    ######## train #########  
    model = run_training(model=model, 
                         train_set=train_set, 
                         test_set=test_set, 
                         criterion=criterion, 
                         optimizer=optimizer, 
                         args=args, 
                         train_set_cond=None, 
                         test_set_cond=None) 


if __name__ == "__main__" :
    main()


