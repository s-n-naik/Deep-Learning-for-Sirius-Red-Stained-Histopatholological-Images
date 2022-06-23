import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from PIL import Image
from Reinhard import Normalizer
from data_augment import RandomAugment
from dataset_generic import CustomDataset, load_dataframe, get_indep_test_sets, MultipleInferenceDataset
from model_zoo import Encoder, get_ozan_ciga, freeze_encoder
from ssc_utils import SSCMaxPoolClassifier, train_model_w_ssc, SSCGatedAttentionClassifier, get_scores_ssc, \
    get_scores_ssc_multiple_inference, train_model_w_ssc_multiple_inference
from training import train_model, train_model_multiple_inference
from utils import get_scores, set_device_and_seed, load_model_weights, get_scores_multiple_inference, get_model, get_model_ssc



def run_inference(configuration):
    print("Configuration for run:", configuration)
    # set random seed and device CPU / GPU
    device = set_device_and_seed(GPU=True, seed=configuration["seed"],
                                 gpu_name=configuration["gpu_name"])
    # info to load the data
    tiles_summary_data = configuration["tile_path"]
    data_dir = configuration["data_dir"]
    num_workers =16
    pin_memory=False
    resize=configuration["resize"]
    img_size = configuration["img_size"]

    # make labels binary
    label_map = {"1a": 0, 
                 "1b": 0, 
                 "1c": 0, 
                 "2": 0,
                 "0": 0,
                 "3":1, 
                 "4":1,
                "1": 0}

    test_df = load_dataframe(os.path.abspath(tiles_summary_data), label_map)
    
    # remove for experiment running
    test = True
    if test:
        print("Testing mode")
        test_df = test_df.head(20)

    # get normalizer
    if configuration["stain_normalization"]["apply_reinhard"]:
        source_dir = configuration["stain_normalization"]["source_dir"]
        normalizer = Normalizer(source_path = source_dir)
    else:
        normalizer = None


    # cv_results
    test_results = {"test_f1": [], "test_accuracy": [], "test_cm": []}

    # create datasets
    if configuration["multiple_inference"]:
        test_dataset = MultipleInferenceDataset(test_df, data_dir,  keep_top=configuration["bag_size"], verbose=False, hsv=configuration["hsv"], cmyk=configuration["cmyk"], transform=None, train=False,  resize=resize, img_size=img_size,normalizer=normalizer)
        print(f"Created MI datasets: Testing on {len(test_dataset)} WSIs")
    else:
        test_dataset = CustomDataset(test_df, data_dir,  keep_top=configuration["bag_size"], verbose=False, hsv=configuration["hsv"], cmyk=configuration["cmyk"], transform=None, train=False,  resize=resize, img_size=img_size,normalizer=normalizer)
        print(f"Created Normal datasets: Testing on {len(test_dataset)} WSIs")

    # create dataloaders
    test_loader = data_utils.DataLoader(test_dataset, batch_size = configuration["batch_size"], shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    print(f"Created Dataloader with num workers {num_workers} and pinned memory {pin_memory}")
    if configuration["use_ssc"]:
        print("Getting SSC model")
        model = get_model_ssc(configuration, device)
    else:
        print("Getting non-SSC model")
        model = get_model(configuration, device)
    print(model)

    # track model weights
    criterion = nn.BCELoss().to(device)
    print(f"Loading best model")
    # reload best model
    if configuration["multi_gpu"]:
        model.module.load_state_dict(torch.load(configuration["best_model_weights_path"]))
    else:
        model.load_state_dict(torch.load(configuration["best_model_weights_path"], map_location=configuration["gpu_name"]))

    if configuration["use_ssc"]:  
        print("Get scores with SSC")
        if configuration["multiple_inference"]:
            cm, f1, accuracy = get_scores_ssc_multiple_inference(model, test_loader, device, count_epochs,  visualise=True)
        else:
            cm, f1, accuracy = get_scores_ssc(model, test_loader, device, count_epochs,  visualise=True)
    else:

        print("Get scores without SSC")
        if configuration["multiple_inference"]:
            cm, f1, accuracy = get_scores_multiple_inference(model, test_loader, device)
        else:
            cm, f1, accuracy = get_scores(model, test_loader, device)

    test_results["test_accuracy"].append(accuracy)
    test_results["test_f1"].append(f1)
    test_results["test_cm"].append(cm)


    # log mean and stdev of cv results
    mean_val_f1 = np.mean(test_results["test_f1"])
    std_val_f1 = np.std(test_results["test_f1"])
    mean_val_acc = np.mean(test_results["test_accuracy"])
    std_val_acc = np.std(test_results["test_accuracy"])

    print('Final results of run on Test set: Accuracy =  {:.2f}%, F1 =  {:.2f}%'.format(accuracy*100, f1*100))

    # get chart of test set distribution
    plt.figure()
    sn.set_style('darkgrid')
    sn.set_palette('Set2')
    data = pd.DataFrame({"Accuracy": test_results["test_accuracy"], "F1 score": test_results["test_f1"]})
    sn.boxplot(data=data)
    plt.savefig("box_and_whisker.jpg")
    plt.show()


    # get total confusion matrix
    total = np.sum(test_results["test_cm"], axis=0)
    plt.figure()
    sn.heatmap(total, annot=True)
    plt.title("Overall confusion matrix across independent test sets")
    plt.savefig("Confusion_matrix_total.jpg")
    plt.show()


if __name__ == "__main__":
    
    configuration = {
        # Training Parameters
        "num_epochs": 1,
        "batch_size": 6,
        "grad_accum_freq": 1, # update gradients every n batches
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "gpu_name": "cuda:0",
        "seed": 42,
        "val_folds": 3, # to get 1 val set for early stopping of size 1/val_folds of the train set
        "test_folds": 3, # num test folds for cross validation
        "multi_gpu": False, # if True, use all available JPUs with DataParallel
        "early_stopping": True, # early stopping of training on val set
        "test_every": 1, # frequency to run inference on val set during training
        
        # DataLoading parameters
        "tile_path":"/data/MSc_students_accounts/sneha/tiles_summary_5x_2.5mm_50mm_adapt_threshold.csv", # abs path to csv file with tiling info
        "data_dir": "/data/goldin/images_raw/nafld_liver_biopsies_sirius_red/", # abs path to loc of .ndpi files
        "multiple_inference": False, # train with multiple inference dataset
        "bag_size": 10, # tiles per bag
        
        
        # Preprocessing parameters
        "resize": True, # resize images before trainnig
        "img_size": 224, # resize to img_size x img_size patches
        "stain_normalization": # apply reinhard stain normalisation based on source image in source_dir
        {
            "apply_reinhard": True, 
            "source_dir":"/data/MSc_students_accounts/sneha/sneha/sirius_red-master/reinhard_source.jpg"
        },
        "transform_color_jitter": None, # hsv color jitter to apply
        "hsv": False, # input img color space
        "cmyk": False,# input img color space
        "transform_kernel_size": 0, # gaussian blur
        "transform_max_rotation": 50, # random rotation
        "transform_min_crop": 1, # random resized crop
        
        # Model Parameters
        "encoder": "se_resnet18", # se_resnet18, se_resnet34, resnet18, resnet34, simclr
        "image_net_pretrained": True, # load imgnet pretrained weights
        "best_model_weights_path": "enter_path_here", # location of pretrained encoder weights to load
        "freeze": False, # True, False or "part" - encoder weights to freeze
        "dropout": False,
        "num_layers": 1, # only for max pool models
        "aggregation": "gated_attention", # gated_attention or simple_attention or max_pool
        
        # SSC module parameters
        "use_ssc": True,
        "ssc_reconst_loss": "l2",
        "ssc_num_routings": 3, # R
        "ssc_lr": 0.1, # initial lr for ssc_module
        "apply_ssc_scheduler": (0.1,30), # Decay by factor of x0.1 every 30 epochs. False if not applying scheduler
        "ssc_num_stains": 2, # S
        "ssc_num_groups": 6, # M
        "ssc_group_width": 3, # N
        "ssc_use_od": True, # apply OD transformation in SSC capsule
        "ssc_in_channels": 3 # num channels in input image (3=RGB/HSV, 4 = CMYK)
    }
    
    run_finetune(configuration)
