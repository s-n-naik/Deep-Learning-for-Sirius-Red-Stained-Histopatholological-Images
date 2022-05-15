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



def run_finetune(configuration):
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

    tiles_summary = load_dataframe(os.path.abspath(tiles_summary_data), label_map)
    
#     # remove for experiment running
#     test = True
#     if test:
#         print("Testing mode")
#         tiles_summary = tiles_summary.head(20)

    # get training transforms
    transform = RandomAugment(crop_scale= (configuration["transform_min_crop"], 1.0), max_rotation=configuration["transform_max_rotation"], kernel_size=configuration["transform_kernel_size"], color=configuration["transform_color_jitter"])

    # get normalizer
    if configuration["stain_normalization"]["apply_reinhard"]:
        source_dir = configuration["stain_normalization"]["source_dir"]
        normalizer = Normalizer(source_path = source_dir)
    else:
        normalizer = None

    # get test and val data
    test_folds, val_folds_list = get_indep_test_sets(tiles_summary, seed=configuration["seed"], n_splits_test=configuration["test_folds"], n_splits_val=configuration["val_folds"])

    # cv_results
    test_results = {"test_f1": [], "test_accuracy": [], "test_cm": []}

    # iterate over val sets to get mean + std sweep results
    for i, (full_train_idx, test_idx) in enumerate(test_folds):
        print(f"Starting test fold {i}")
        best_acc = 0
        best_epoch = 0
        full_train_df = tiles_summary.iloc[full_train_idx]
        test_df = tiles_summary.iloc[test_idx]
        train_idx, val_idx = val_folds_list[i][0] # get a single val set per test fold
        train_df = full_train_df.iloc[train_idx]
        val_df = full_train_df.iloc[val_idx]
        # create datasets
        if configuration["multiple_inference"]:
            train_dataset = MultipleInferenceDataset(train_df,data_dir,  keep_top=configuration["bag_size"], verbose=False,hsv=configuration["hsv"], cmyk=configuration["cmyk"] , transform=transform, train=True,  resize=resize, img_size=img_size, normalizer=normalizer)
            val_dataset = MultipleInferenceDataset(val_df, data_dir,  keep_top=configuration["bag_size"], verbose=False, hsv=configuration["hsv"], cmyk=configuration["cmyk"], transform=None, train=False,  resize=resize, img_size=img_size, normalizer=normalizer)
            test_dataset = MultipleInferenceDataset(test_df, data_dir,  keep_top=configuration["bag_size"], verbose=False, hsv=configuration["hsv"], cmyk=configuration["cmyk"], transform=None, train=False,  resize=resize, img_size=img_size,normalizer=normalizer)
            print(f"Created MI datasets: Training on {len(train_dataset)} WSIs, Validating on {len(val_dataset)} WSIs and Testing on {len(test_dataset)} WSIs")
        else:
            train_dataset = CustomDataset(train_df,data_dir,  keep_top=configuration["bag_size"], verbose=False,hsv=configuration["hsv"], cmyk=configuration["cmyk"] , transform=transform,train=True,  resize=resize, img_size=img_size, normalizer=normalizer)
            val_dataset = CustomDataset(val_df, data_dir,  keep_top=configuration["bag_size"], verbose=False, hsv=configuration["hsv"], cmyk=configuration["cmyk"], transform=None, train=False,  resize=resize, img_size=img_size, normalizer=normalizer)
            test_dataset = CustomDataset(test_df, data_dir,  keep_top=configuration["bag_size"], verbose=False, hsv=configuration["hsv"], cmyk=configuration["cmyk"], transform=None, train=False,  resize=resize, img_size=img_size,normalizer=normalizer)
            print(f"Created Normal datasets: Training on {len(train_dataset)} WSIs, Validating on {len(val_dataset)} WSIs and Testing on {len(test_dataset)} WSIs")
        
        # create dataloaders
        train_loader = data_utils.DataLoader(train_dataset, batch_size = configuration["batch_size"], shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = data_utils.DataLoader(val_dataset, batch_size = configuration["batch_size"], shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = data_utils.DataLoader(test_dataset, batch_size = configuration["batch_size"], shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

        print(f"Created Dataloaders with num workers {num_workers} and pinned memory {pin_memory}")
        if configuration["use_ssc"]:
            print("Getting SSC model")
            model = get_model_ssc(configuration, device)
        else:
            print("Getting non-SSC model")
            model = get_model(configuration, device)
        print(model)

        # track model weights
        criterion = nn.BCELoss().to(device)

        ssc_params = [param for name, param in model.named_parameters() if ("ssc_module" in name) and (param.requires_grad)]
        other_params = [param for name, param in model.named_parameters() if ("ssc_module" not in name) and (param.requires_grad)]
        print("Parameters in SSC module", len(ssc_params))
        print("Parameters not in SSC module", len(other_params))
        if configuration["use_ssc"]:
            optimizer = torch.optim.Adam([{'params':ssc_params, 'lr': configuration["ssc_lr"]}, {'params':other_params}], lr=configuration["learning_rate"],  weight_decay=configuration["weight_decay"])
        else:
            print("Using optimiser with one lr", configuration["learning_rate"])
            optimizer = optim.Adam(other_params, lr=configuration["learning_rate"], weight_decay=configuration["weight_decay"])


        if configuration["use_ssc"] and configuration["apply_ssc_scheduler"]:
            decay_factor, decay_every = configuration["apply_ssc_scheduler"]
            print(f"Applying Scheduler to SSC unit that decays from {configuration['ssc_lr']} by {decay_factor} every {decay_every} steps")
            print(f"Scheduler leaves encoder lr constant at {configuration['learning_rate']}")
            lambda1 = lambda epoch: decay_factor ** (epoch // decay_every)
            lambda2 = lambda epoch: epoch**0
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])

        else:
            scheduler=None

        test_accuracy_progress = []
        test_f1_progress = []
        test_accuracy = []
        test_f1 = []
        test_cm = []
        train_accuracy = []
        train_loss = []
        val_accuracy = []

        num_epochs_train = configuration["test_every"]

        # start training
        count_epochs = 0
        while count_epochs <= (configuration["num_epochs"]-num_epochs_train):
            print(f"Training epochs {count_epochs} to {count_epochs + num_epochs_train} for fold {i}")
            start_epoch = count_epochs
            count_epochs += num_epochs_train
            if configuration["use_ssc"]:
                print("Training with SSC")
                if configuration["multiple_inference"]:
                    print("Training with multiple inference")
                    plotting_dict_train = train_model_w_ssc_multiple_inference(model, optimizer, train_loader, device, criterion, num_epochs_train, start_epoch, verbose=True, reconst_mode=configuration["ssc_reconst_loss"], scheduler=scheduler, visualise=False, accum_iter=configuration["grad_accum_freq"])

                else:
                    plotting_dict_train = train_model_w_ssc(model, optimizer, train_loader, device, criterion, num_epochs_train, start_epoch, verbose=True, reconst_mode=configuration["ssc_reconst_loss"], scheduler=scheduler, visualise=False, accum_iter=configuration["grad_accum_freq"])
            else:
                print("Training without SSC")
                if configuration["multiple_inference"]:
                    print("Training with multiple inference")
                    plotting_dict_train = train_model_multiple_inference(model, optimizer, train_loader, device, criterion, num_epochs_train, verbose=True, scheduler=scheduler, accum_iter=configuration["grad_accum_freq"])
                else:
                    plotting_dict_train = train_model(model, optimizer, train_loader, device, criterion, num_epochs_train,verbose=True, scheduler=scheduler, accum_iter=configuration["grad_accum_freq"], weighted_loss=False)

            train_accuracy.extend(plotting_dict_train["accuracy"])
            train_loss.extend(plotting_dict_train["loss"])


            # get intermediate results on test set or val set to display only
            print("Getting intermediate results on Val Set")
            
            if configuration["use_ssc"]:  
                print("Get scores with SSC")
                if configuration["multiple_inference"]:
                    cm, f1, accuracy = get_scores_ssc_multiple_inference(model, val_loader, device, count_epochs, visualise=True)
                else:
                    cm, f1, accuracy = get_scores_ssc(model, val_loader, device, count_epochs, visualise=True)
            else:

                print("Get scores without SSC")
                if configuration["multiple_inference"]:
                    cm, f1, accuracy = get_scores_multiple_inference(model, val_loader, device)
                else:
                    cm, f1, accuracy = get_scores(model, val_loader, device)

            test_accuracy_progress.append(accuracy)
            test_f1_progress.append(f1)

            if accuracy > best_acc:
                best_acc = accuracy
                best_epoch = count_epochs
                print(f"Updating best accuracy to {best_acc}")
                print(f"Updating best model at epoch {count_epochs}")

                if configuration["multi_gpu"]:
                    torch.save(model.module.state_dict(), f"best_model_fold_{i}_{configuration['gpu_name']}.h5")
                else:
                    torch.save(model.state_dict(), f"best_model_fold_{i}_{configuration['gpu_name']}.h5")

            if configuration["early_stopping"]:
                print(f"Checking whether to stop training criteria is met manual at epoch {count_epochs}, last updated best accuracy at epoch {best_epoch}")
                # if more than 30 epochs of model training have passed and performance is still below the best historic performance, stop training
                if (count_epochs > 30) and (count_epochs - best_epoch > 15):
                    print("Stopping training")
                    # exit while loop
                    break



        print(f"Finished {count_epochs} training epochs, reloading best model")
        # reload best model
        if configuration["multi_gpu"]:
            model.module.load_state_dict(torch.load(f"best_model_fold_{i}_{configuration['gpu_name']}.h5"))
        else:
            model.load_state_dict(torch.load(f"best_model_fold_{i}_{configuration['gpu_name']}.h5", map_location=configuration["gpu_name"]))

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

        # plot training accuracy at end of training
        plt.figure()
        plt.plot(np.arange(0, len(train_accuracy), 1), train_accuracy, label="train accuracy")
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend()
        plt.savefig(f"training_acc_fold_{i}.jpg")
        plt.show()


    # log mean and stdev of cv results
    mean_val_f1 = np.mean(test_results["test_f1"])
    std_val_f1 = np.std(test_results["test_f1"])
    mean_val_acc = np.mean(test_results["test_accuracy"])
    std_val_acc = np.std(test_results["test_accuracy"])

    print('Final results of run on Test set: Accuracy =  {:.2f} +/- {:.2f} stdev, F1 =  {:.2f} +/- {:.2f} stdev'.format(mean_val_acc, std_val_acc, mean_val_f1, std_val_f1))

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
        "load_path": False, # if not False, location of pretrained encoder weights to load
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
