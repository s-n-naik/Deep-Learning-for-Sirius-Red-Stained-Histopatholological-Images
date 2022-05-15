import os
import numpy as np
from torchvision import transforms

    
class RandomAugment:
    
    def __init__(self, crop_scale= (0.08, 1.0), max_rotation=90, kernel_size=1, color=None, verbose=False, p=0.5):
        '''
        Random apply 
    1) RandomResizedCrop: If crop scale is not False
    2) Horiz and Vertical Flips
    3) Rotation - random between - max_rotation and + max rotation, fill color = mean hsv
    4) Gaussian Blur - kernel size must be an odd integer
    5) If color is not None ('strong', 'medium', 'light') color jitter is also applied
    6) p: probability with which transforms are applied, default = 0.5
        '''
        self.crop_scale = crop_scale
        self.max_rotation = max_rotation
        self.kernel_size = kernel_size
        self.color=color
        self.verbose= verbose
        self.p = p
        

    def __call__(self, img):
        '''
        Random apply:
            1) RandomResizedCrop
            2) Horiz and Vertical Flips
            3) Rotation - random between - max_rotation and + max rotation, fill color = mean hsv
            4) Gaussian Blur - kernel size
            5) Color jitter - none, strong - weak
        '''
        augmented_img = img
        img_size = img.size[0]
        r, g, b = img.split()
        fillcolor = (int(np.round(np.mean(r),0)),int(np.round(np.mean(g),0)),int(np.round(np.mean(b),0)))
        
        flips = transforms.Compose([transforms.RandomHorizontalFlip(p=self.p),
                             transforms.RandomVerticalFlip(p=self.p)])
        
        augmented_img = flips(augmented_img)
        
        if self.crop_scale[0] != 1:
            # if min crop is not 1, then append the random resized crop transform
            augmented_img = transforms.RandomResizedCrop(img_size, scale=self.crop_scale)(augmented_img)
            
        if self.kernel_size != 0:
            # if kernel size is > 0 then check it is odd then append the blur transformation
            self.kernel_size = int(self.kernel_size)
            if self.kernel_size % 2 == 0:
                self.kernel_size = self.kernel_size + 1
                
            augmented_img = transforms.ToTensor()(augmented_img)
            
            # Gaussian Blur
            blur = transforms.RandomApply([transforms.GaussianBlur(self.kernel_size, sigma=1)], p=self.p) #0.1,2.0
            l2_before = augmented_img.pow(2).sum().sqrt()
            augmented_img = blur(augmented_img)
            l2_after = augmented_img.pow(2).sum().sqrt()
            
            # scaling image to ensrue L2 norm is same before and after blurring
            augmented_img = augmented_img * l2_before / l2_after
            augmented_img = transforms.ToPILImage()(augmented_img)
            
        if self.max_rotation > 0:
            # if a rotation degree > 0 has been selected, append the random rotation transform
            augmented_img = transforms.RandomRotation(self.max_rotation, fill=fillcolor)(augmented_img)
        
        # apply HSV colorjitter
        if self.color == "strong":
            color_jitter = transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.4)], p=self.p)
        elif self.color == 'medium':
            color_jitter = transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=self.p)
        elif self.color == 'light':
            color_jitter = transforms.RandomApply([transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)], p=self.p)
        else:
            color_jitter = None
        
        if self.verbose:
            print("Applying color transforms:", color_jitter)
        
        if color_jitter is not None:
            augmented_img = color_jitter(augmented_img)
        
        return augmented_img
    
    

    
def reinhard(tiles, normalizer):
    '''
    helper function to apply reinhard stain normalisation - first need to split into individual tiles
    '''
    n_tiles = tiles.shape[0]
    for i in range(n_tiles):
        tiles[i] = normalizer.transform(np.array(tiles[i]))
    return tiles
