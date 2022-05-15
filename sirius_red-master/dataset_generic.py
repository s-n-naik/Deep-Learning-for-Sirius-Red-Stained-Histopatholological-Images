import os
import random
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
import torch
import torch.utils.data as data_utils
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms
from torchvision.utils import make_grid
from ndpi_slide import Slide

class CustomDataset(data_utils.Dataset):
    def __init__(self, tile_df,
                 original_data_dir,
                 transform = None,
                 keep_top=False,
                 random_bag_size=False,
                 seed=1,
                 resize=True,
                 img_size=224,
                 hsv=False,
                 cmyk=False,
                 lab=False,
                 train=True,
                 normalizer = None,
                 verbose=False):
        '''
        Dataloader 1: apply transformations to tiles and return MIL bags + bag label
        
        :param tile_df: pandas dataframe with info for loading tiles for WSIs in dataset
        :param original_data_dir: loc of ndpi files
        :param transform: torchvision transforms to be applied to PIL images, set to None if train=False, default=None
        :param keep_top: number of tiles to keep per bag. If there are more than keep_top in the bag, then selected based on highest % red pixels, if there are less than keep_top, the additional numbers are made up by applying further data augmentation
        :param random_bag_size: If not false, number of tiles to keep per bag. Randomly selects this number of tiles for WSI, default = False.
        :param seed: random state
        :param resize: resize images to img_size
        :param img_size: int size in pixels to resize square patches to, default = 224
        :param hsv: If True, image patches are converted to HSV mode before dataloading, default False
        :param cmyk: If True, image patches are converted to CMYK mode before dataloading, default False
        :param lab: If True, image patches are converted to LAB mode before dataloading, default False
        :param train: bool, true if the data is train data (transforms will be applied), false if val / test data (no transforms applied)
        :param normalizer: Normalizer class instance if Reinhard Stain Normalisation is to be applied, default = None.
        :param verbose: how much to print, default=False
        
        '''
        
        self.tiles_summary = tile_df
        self.verbose = verbose
        self.keep_top = keep_top
        self.random_bag_size = random_bag_size
        self.data_dict = original_data_dir
        
        self.train = train
        self.resize = resize
        self.img_size = img_size
        self.hsv = hsv
        self.cmyk = cmyk
     
        self.normalizer = normalizer 
        
        
        if self.train:
            self.transform = transform
        else:
            self.transform = None
        
        self.seed = seed
        self.set_seeds()
        
        self.bag_stats = dict()
        self._get_bag_stats()
     
    def _get_bag_stats(self):
        '''
        Generates jpg histogram of tiles per bag in dataset for each class
        '''
        # filter based on class / joint score
        category = "stage"
        hist_list = []
        labels_list = []
        for c in list(np.unique(self.tiles_summary[category])):
            labels_list.append(c)
            number_tiles_per_bag = self.tiles_summary[self.tiles_summary[category]==c]["mostly_tissue"]
            hist_list.append(number_tiles_per_bag)
            per_class_stats = (round(np.mean(number_tiles_per_bag)), round(np.std(number_tiles_per_bag)))
            self.bag_stats[c] = per_class_stats
            if self.verbose:
                print(f"Class {c}: Mean tiles / bag = {per_class_stats[0]}, Std tiles / bag = {per_class_stats[1]}")

        if self.verbose:       
            plt.figure()
            plt.hist(hist_list,stacked=True,histtype='bar', label=labels_list)
            plt.xlabel("Tiles per bag")
            plt.ylabel("Frequency")
            plt.legend()
            plt.savefig("bag_stats.jpg")
            plt.show()
            
            
    
        
    
    def set_seeds(self): 
        '''
        Set seeds for reproducibility
        
        '''
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        
    
    def _construct_bag(self, index):
        '''
        Gets gets row from data frame according to index
        Collects tiles
        Resizes tiles
        Applies any transforms required to each tile in bag in PIL format
        Converts to tensor
        returns: bag (list of tensors), bag label 
        '''
        starttime = time.time()
        # Get info from df
        wsi_info = self.tiles_summary.iloc[index]
        stage = int(wsi_info["stage"])
        path = wsi_info["ndpi_file"]
        num_tiles = int(wsi_info["mostly_tissue"])
        red_info_list = wsi_info["red_info"]
        
        
        if self.verbose:
            # use the Slide class to print thumbnail / stats on slide
            slide = Slide(os.path.abspath(path), self.verbose, self.verbose)
            slide = slide.slide
            print("Stage: ", stage)
            print("Num tiles in bag", num_tiles)
            
        else:
            slide= openslide.open_slide(os.path.abspath(path))
        
        tile_list = wsi_info.dropna()
       
        # get only the columns which contain the tile info
        keep_list = []
        for index in tile_list.index:
            try:
                col = int(index)
                keep_list.append(index)
            except:
                continue
        tile_list = tile_list[keep_list].tolist()
        info_list = []
        for tile in tile_list:
            info = re.split("/", tile)
            level = int(info[0])
            location = (int(info[1]), int(info[2]))
            size = (int(info[3]), int(info[4]))
            red_percent = float(info[5])
            info_list.append((level, location, size, red_percent))
            
        # info for tiles sorted largest to smallest in terms of red pixels
        sorted_info_list = sorted(info_list, key= lambda x:x[-1], reverse=True)
        
        try:
            assert len(sorted_info_list) > 0, f"For index {index}, there are no tiles"
        except AssertionError:
            print(f"For index {index}, there are no tiles")
        
        # get top N tiles in terms of red pixels
        if self.keep_top: 
            if len(sorted_info_list) > self.keep_top:
                final_tile_list = sorted_info_list[:self.keep_top]
            elif len(sorted_info_list) == self.keep_top:
                final_tile_list = sorted_info_list
            else:
                extra_tiles_needed = self.keep_top - len(info_list)
                final_tile_list = sorted_info_list + random.choices(sorted_info_list,k=extra_tiles_needed)
    
        # Randomly choose N tiles (not condsidering % red)
        elif self.random_bag_size:
            final_tile_list = random.choices(info_list, k=self.random_bag_size) 
        else:
            final_tile_list = sorted_info_list
        
        # Read patches from original ndpi raw file using OpenSlide
        bag = []
        for level, location, size, red_percent in final_tile_list:
            
            img = slide.read_region(location = location, level=level, size=size).convert("RGB")
            
            # Apply transforms to tiles depending on args in dataset
            if self.resize:
                img = img.resize((self.img_size, self.img_size))
            if self.normalizer is not None:
                img = self.normalizer.transform(np.array(transforms.ToTensor()(img)))
                img = transforms.ToPILImage()(img).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)

            if self.verbose:
                print(f"Level {level}, Location {location}, Size {size}, {slide.level_dimensions[level]}")
                plt.figure()
                plt.imshow(img)
                plt.show()
                
            if self.hsv:
                img = img.convert("HSV")
            elif self.cmyk:
                img = img.convert("CMYK")
            
            # convert to tensor for dataloader
            img = transforms.ToTensor()(img) #.unsqueeze_(0)
            bag.append(img.unsqueeze(0))
            
        slide.close()
        
        if self.verbose:    
            print("bag length", len(bag), "stage", stage)
            print(f"_construct_bag: {time.time() - starttime}s")
        return bag, stage
        
    def __len__(self):
        return len(self.tiles_summary)
    
    def __getitem__(self, index):
        bag, label = self._construct_bag(index)
        bag_tensor = torch.cat(bag, axis=0)
        # tensor of shape [1, bag_size, num_channels, img_size, img_size]
        return bag_tensor, label


    
    

class CustomPairDataset(data_utils.Dataset):
    def __init__(self, tile_df,
                 original_data_dir,
                 transform = None,
                 bag_size = 1,
                 use_red=False,
                 seed=1,
                 resize=True,
                 img_size=224,
                 verbose=False):
        '''
        Custom dataset class for unsupervised pretraining
        :param tile_df: pandas dataframe with info for loading tiles for WSIs in dataset
        :param original_data_dir: loc of ndpi files
        :param transform: RandomAugment torchvision transforms to be applied to PIL images, set to None if train=False, default=None
        :param bag size: number of tiles to keep per bag. Randomly samples.
        :param use_red: sample tiles for training based on % red pixels, default = False
        :param seed: random state
        :param resize: resize images to img_size
        :param img_size: int size in pixels to resize square patches to
        :param verbose: how much to print, default=False (not printing)
        '''
        
        self.tiles_summary = tile_df
        self.verbose = verbose
        self.data_dict = original_data_dir
        self.use_red = use_red
        self.bag_size = bag_size
        
        self.resize = resize
        self.img_size = img_size
        
        self.transform = transform
        
        self.seed = seed
        self.set_seeds()
        
        if self.verbose:
            self._get_bag_stats()
     
    
    def _get_bag_stats(self):
        '''
        Generates jpg histogram of tiles per bag in dataset for each class
        
        '''
        number_tiles_per_bag = self.tiles_summary["mostly_tissue"]
        if self.verbose:
            plt.figure()
            plt.hist(number_tiles_per_bag)
            plt.xlabel("Tiles per bag")
            plt.ylabel("Frequency")
            plt.savefig("bag_stats.jpg")
            plt.show()
            print("Min", min(number_tiles_per_bag),"Max", max(number_tiles_per_bag),"Mean", np.mean(number_tiles_per_bag), "Std", np.std(number_tiles_per_bag))

        
    
    def set_seeds(self): 
        '''
        Set random seeds for reproducibility
        
        '''
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        
    
    def _construct_pair(self, index):
        '''
        Gets gets row from data frame according to index
        Collects tiles
        Resizes tiles if required
        Samples a tile at random from the bag
        Applies TWO random transforms to tile to get a pair of images with diff transforms
        returns: tile_1, tile_2, bag label 
        '''
        # get slide info
        wsi_info = self.tiles_summary.iloc[index]
        stage = wsi_info["stage"]
        path = wsi_info["ndpi_file"]
        num_tiles = int(wsi_info["mostly_tissue"])
        red_info_list = wsi_info["red_info"]
        
        if self.verbose:
            # use the Slide class to print thumbnail / stats on slide
            slide = Slide(os.path.abspath(path), self.verbose, self.verbose)
            slide = slide.slide
            print("Stage: ", stage)
            print("Num tiles in bag", num_tiles)
            
        else:
            slide= openslide.open_slide(os.path.abspath(path))
        
        tile_list = wsi_info.dropna()
        # get only the columns which contain the tile info
        keep_list = []
        for index in tile_list.index:
            try:
                col = int(index)
                keep_list.append(index)
            except:
                continue
        tile_list = tile_list[keep_list].tolist()
        
        
        info_list = []
        for tile in tile_list:
            info = re.split("/", tile)
            level = int(info[0])
            location = (int(info[1]), int(info[2]))
            size = (int(info[3]), int(info[4]))
            red_percent = float(info[5])
            info_list.append((level, location, size, red_percent))
            
        sorted_info_list = sorted(info_list, key= lambda x:x[-1], reverse=True)
        try:
            assert len(sorted_info_list) > 0, f"For index {index}, there are no tiles"
        except AssertionError:
            print(f"For index {index}, there are no tiles")
        
        # sample tiles for pretraining
        if self.use_red: 
            # random choice weighted by number of red pixels 
            red_pixel_list = [x[-1] for x in sorted_info_list]
            normalise = np.sum(red_pixel_list)
            red_pixel_probabilities = [x/normalise for x in red_pixel_list]
            # select a tile at random to sample for learning
            final_tile_list = random.choices(sorted_info_list, k=self.bag_size, weights=red_pixel_probabilities)
             
        else:
            # random chocie of tiles
            final_tile_list = random.choices(sorted_info_list, k=self.bag_size)
        
        # apply two separate transforms
        bag_1 = []
        bag_2 = []
        for level, location, size, red_percent in final_tile_list:
            
            img = slide.read_region(location = location, level=level, size=size).convert("RGB")
            if self.resize:
                img = img.resize((self.img_size, self.img_size))
            if self.transform is not None:
                img_1 = self.transform(img)
                img_2 = self.transform(img)
                
                
            img_1 = transforms.ToTensor()(img_1)
            img_2 = transforms.ToTensor()(img_2)

            if self.verbose:
                print(f"Level {level}, Location {location}, Size {size}, {slide.level_dimensions[level]}")
                f, (ax1, ax2) = plt.subplots(1,2)
                ax1.imshow(img_1.permute(2,1,0))
                ax2.imshow(img_2.permute(2,1,0))
                plt.show()

            # convert to tensor for dataloader
            bag_1.append(img_1.unsqueeze(0))
            bag_2.append(img_2.unsqueeze(0))
            
        slide.close()
        
        if self.verbose:    
            print("bag lengths", len(bag_1), "stage", stage)
        
        return bag_1, bag_2, stage
        
    def __len__(self):
        return len(self.tiles_summary)
    
    def __getitem__(self, index):
        bag1, bag2, label = self._construct_pair(index)
        bag_tensor_1, bag_tensor_2 = torch.cat(bag1, axis=0), torch.cat(bag2, axis=0)
        return bag_tensor_1, bag_tensor_2, label
    
    
class MultipleInferenceDataset(data_utils.Dataset):
    def __init__(self, tile_df,
                 original_data_dir,
                 transform = None,
                 keep_top=10,
                 seed=1,
                 resize=True,
                 img_size=224,
                 hsv=False,
                 cmyk=False,
                 train=False,
                 normalizer = None,
                 verbose=False):
        '''
        DataLoader 2 (Multiple Inference) to apply transformations to tiles and return MIL bags + bag label
        :param tile_df: pandas dataframe with info for loading tiles for WSIs in dataset
        :param original_data_dir: loc of ndpi files
        :param transform: torchvision transforms to be applied to PIL images, set to None if train=False, default=None
        :param keep_top: number of tiles to use per bag
        :param seed: random state
        :param resize: resize images to img_size
        :param img_size: int size in pixels to resize square patches to, default = 224
        :param hsv: If True, image patches are converted to HSV mode before dataloading, default False
        :param cmyk: If True, image patches are converted to CMYK mode before dataloading, default False
        :param train: bool, true if the data is train data (transforms will be applied), false if val / test data (no transforms applied)
        :param normalizer: Normalizer class instance if Reinhard Stain Normalisation is to be applied, default = None.
        :param verbose: how much to print, default=False
        
        '''
        
        self.tiles_summary = tile_df
        self.verbose = verbose
        self.bag_size = keep_top
        self.data_dict = original_data_dir
        self.train = train
        self.resize = resize
        self.img_size = img_size
        self.hsv = hsv
        self.cmyk = cmyk
        self.normalizer = normalizer 
        
        
        if self.train:
            self.transform = transform
        else:
            self.transform = None
        
        self.seed = seed
        self.set_seeds()
        
        self.bag_stats = dict()
        self._get_bag_stats()
        self._form_bags()
     
    def _get_bag_stats(self):
        '''
        Generates jpg histogram of tiles per bag in dataset for each class
        '''
        # filter based on class / joint score
        category = "stage"
        
        hist_list = []
        labels_list = []
        for c in list(np.unique(self.tiles_summary[category])):
            labels_list.append(c)
            number_tiles_per_bag = self.tiles_summary[self.tiles_summary[category]==c]["mostly_tissue"]
            hist_list.append(number_tiles_per_bag)
            per_class_stats = (round(np.mean(number_tiles_per_bag)), round(np.std(number_tiles_per_bag)))
            self.bag_stats[c] = per_class_stats
            if self.verbose:
                print(f"Class {c}: Mean tiles / bag = {per_class_stats[0]}, Std tiles / bag = {per_class_stats[1]}")

        if self.verbose:       
            plt.figure()
            plt.hist(hist_list,stacked=True,histtype='bar', label=labels_list)
            plt.xlabel("Tiles per bag")
            plt.ylabel("Frequency")
            plt.legend()
            plt.savefig("bag_stats.jpg")
            plt.show()
            
            
    def _form_bags(self, stage_df=None, n=None):
        '''
        Function takes all valid tiles in tiles_summary dataframe for each WSI
        Randomly shuffles each set of tiles and splits into as many bags without replacement of size 'self.keep_top'
        Final bag sampled with replacement to make up remainder
        
        '''
        self.bags = []
        for i in range(len(self.tiles_summary)):
            wsi_info = self.tiles_summary.iloc[i]
            info_list = []
            path = wsi_info["ndpi_file"]
            path_id = int(i)
            label = int(wsi_info["stage"])
            tile_list = wsi_info.dropna()

            # get only the columns which contain the tile info
            keep_list = []
            for index in tile_list.index:
                try:
                    col = int(index)
                    keep_list.append(index)
                except:
                    continue
            tile_list = tile_list[keep_list].tolist()
            for tile in tile_list:
                info = re.split("/", tile)
                level = int(info[0])
                location = (int(info[1]), int(info[2]))
                size = (int(info[3]), int(info[4]))
                red_percent = float(info[5])
                info_list.append((path, level, location, size, red_percent))

            # without replacement generate as many bags as you can
            random.shuffle(info_list)
            bag_size = self.bag_size
            # get all bags
            num_bags = int(len(info_list) // bag_size)
            remainder = int(len(info_list) % bag_size)
            count= 0
            for i in range(num_bags):
                end = count + bag_size
                self.bags.append((path_id, label, info_list[count:end]))
                count = end
            
            # make last bag
            if remainder > 0:
                extra_tiles_needed = self.bag_size - remainder
                final_tile_list = info_list[count:] + random.choices(info_list,k=extra_tiles_needed)
                self.bags.append((path_id, label,final_tile_list))
                

        
    
    def set_seeds(self): 
        '''
        Set seeds for reproducibility
        '''
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        
    def get_tile(self, info):
        '''
        Read + Apply transforms to a tile based on WSI info from tiles summary
        '''
        path, level, location, size, red_percent = info
        slide= openslide.open_slide(os.path.abspath(path))
        img = slide.read_region(location = location, level=level, size=size).convert("RGB")

        if self.resize:
            img = img.resize((self.img_size, self.img_size))

        if self.normalizer is not None:
            img = self.normalizer.transform(np.array(transforms.ToTensor()(img)))
            img = transforms.ToPILImage()(img).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        if self.verbose > 1:
            print(f"Level {level}, Location {location}, Size {size}, {slide.level_dimensions[level]}")

            plt.figure()
            plt.imshow(img)
            plt.show()
        
        if self.hsv:
            img = img.convert("HSV")
        elif self.cmyk:
            img = img.convert("CMYK")
            
        # convert to tensor for dataloader
        img = transforms.ToTensor()(img) #.unsqueeze_(0)
        
        slide.close()
        
        return img.unsqueeze_(0)
    
    
    
    def _construct_bag(self, index):
        '''
        Collects tiles for each bag generated
        Resizes tiles if required
        Applies any transforms required to each tile in bag in PIL format
        Converts to lsit of tensor
        returns: bag (list of tensors), bag label 
        '''

        path, label, info = self.bags[index]
        bag = []

        for tile in info:
            img = self.get_tile(tile)
            bag.append(img)

        return bag, label, path
        
    def __len__(self):
        return len(self.bags)
    
    def __getitem__(self, index):
        starttime = time.time()
        bag, label, path = self._construct_bag(index)
        bag_tensor = torch.cat(bag, axis=0)
        if self.verbose:    
            print("bag length", len(bag), "stage", label, "ndpi", path, sep="\n")
            print(f"_construct_bag: {time.time() - starttime}s")
            # visualise bag
            plt.figure()
            img = make_grid([bag_tensor[i] for i in range(bag_tensor.shape[0])], nrow=self.bag_size)
            npimg = img.cpu().numpy()
            plt.imshow(np.transpose(npimg, (1,2,0)))
            plt.show()
        return bag_tensor, (label, path)    
    
    
def load_dataframe(tiles_summary_data, label_map):
    '''
    Processes + cleans up CSV file with tile data in it
    :param tiles_summary_data: path to csv files with tile data
    :param label_map: dict to change labelling to binary etc
    :return: cleaned dataframe for tiles summary data
    '''
    # DATA PROCESSING
    
    tiles_summary = pd.read_csv(os.path.abspath(tiles_summary_data)).dropna(subset=["0"])
    print(f"There are {len(tiles_summary)} files in {tiles_summary_data}")
    # remove lines where there are no tiles
    no_tiles = tiles_summary[tiles_summary["mostly_tissue"]==0]
    print(f"Removing {len(no_tiles)} WSI with no tiles: {no_tiles['ndpi_file']}")
    tiles_summary = tiles_summary[tiles_summary["mostly_tissue"]!=0]
    print(f"There are {len(tiles_summary)} files in {tiles_summary_data}")
    # make the red info into a list from string
    tiles_summary["red_info"] = tiles_summary["red_info"].astype(str).apply(lambda x: x.strip("[]").split(","))
    # convert stage into binary labels if provided
    if label_map:
        tiles_summary["stage"] = tiles_summary["stage"].apply(lambda x: label_map[x])
    # change to a joint score so that characteristic is taken into account for splitting into test set/ folds
    tiles_summary["joint_score"] = tiles_summary["stage"].map(str)+ tiles_summary["color_characteristics"]
    # files labelled with this character have artifacts/ strange characteristics and so are removed here
    character_e = tiles_summary[tiles_summary["color_characteristics"]=="e"]["ndpi_file"]
    tiles_summary = tiles_summary[~tiles_summary["ndpi_file"].isin(character_e)]
    print(f"There are {len(tiles_summary)} files after removing characteristic 'e' tiles")
    character_c = tiles_summary[tiles_summary["color_characteristics"]=="c"]["ndpi_file"]
    tiles_summary = tiles_summary[~tiles_summary["ndpi_file"].isin(character_c)]
    print(f"There are {len(tiles_summary)} files after removing characteristic 'c' tiles")
    return tiles_summary


def cv_split(tiles_summary, label_map=None, seed=1, test_size = 0.2, n_folds=10):
    '''
    OLD - only train / val split
    :param tiles_summary: cleaned dataframe for full dataset
    :param seed: seed to set random state
    :param test_size: test set size
    :param n_folds: number of folds to split train set into for cross_validation
    :return: cv_fold (generator which splits the train set into 5 folds for train, val), train_set (dataframe with info for full train set), \
    test_set (dataframe with info for test_set) 
    '''
    
    # labels for stratified split based on characteristic + class labels
    labels_list = tiles_summary["joint_score"].tolist()
    print("Label frequency distribution", [(x, labels_list.count(x)) for x in set(labels_list)])
    
    
    # Split into train / test set
    # fudge split since there is only 1 file that is 1d so we can't use stratified shuffle split on it so use kfold for all
    n_splits_test = int(round(1/test_size))
    splitter_test = StratifiedKFold(n_splits=n_splits_test, shuffle=True, random_state=seed)
    
    folds = list(splitter_test.split(np.array(labels_list), np.array(labels_list)))
    
    train_idx, test_idx = folds[0]
    
    
    train_set = tiles_summary.iloc[train_idx]
    test_set = tiles_summary.iloc[test_idx]
    
    splitter_val = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv_folds = list(splitter_val.split(np.array(train_set["joint_score"].tolist()), np.array(train_set["joint_score"].tolist())))
    
    return cv_folds, train_set, test_set
    


def get_indep_test_sets(tiles_summary, seed=1, n_splits_test=5, n_splits_val=10):
    '''
    Train/Val/Test split using StratifiedKFold so n_folds=5 = 20% test set etc. 
    :param tiles_summary: cleaned dataframe for full dataset
    :param seed: seed to set random state
    :param n_splits_test: number of independent test sets (folds) to create
    :return: test_folds (generator for n_splits_test indep test sets), val_folds_list (list of generators for n_splits_val train/val sets, for each of the n_splits_test indep train/test sets)
    
    NOTE: Imperfect splits if not enough entries of each class will give a warning
    
    '''
    
    # labels for stratified split based on characteristic + class labels
    labels_list = tiles_summary["joint_score"].tolist()
    print("Overall Label frequency distribution", [(x, labels_list.count(x)) for x in set(labels_list)])
    
    # Split into train / test set
    splitter_test = StratifiedKFold(n_splits=n_splits_test, shuffle=True, random_state=seed)
    
    test_folds = list(splitter_test.split(np.array(labels_list), np.array(labels_list)))
    
    # split each train set into train/ val set
    splitter_val = StratifiedKFold(n_splits=n_splits_val, shuffle=True, random_state=seed)
    
    val_folds_list = [] # stores n_splits_val - fold generator for each test fold generated
    for i, (train_idx, val_idx) in enumerate(test_folds):
        train_df = tiles_summary.iloc[train_idx]
        test_df = tiles_summary.iloc[val_idx]
        train_labels_list = train_df["joint_score"].tolist()
        print(f"Train label frequency distribution for test fold {i}", [(x, train_labels_list.count(x)) for x in set(train_labels_list)])
        val_folds = list(splitter_val.split(np.array(train_labels_list), np.array(train_labels_list)))
        val_folds_list.append(val_folds)
        
    return test_folds, val_folds_list

