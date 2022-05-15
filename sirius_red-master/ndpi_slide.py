'''
Adapted from base code from Jazz Mack-Smith https://github.ic.ac.uk/jms3/sirius_red
Functions significantly changed or added:
- get_tile_dimensions
- get_valid_tile_mask_ilastik (added)
- get_percent_red (added)
- tile_and_save_jpg_ilastik (added)
- convert_pixels_to_mm (added)
'''


import collections
import os
import re

import PIL
import matplotlib.pyplot as plt
import numpy as np
import openslide
import skimage.filters as sk_filters
from IPython.display import display


class Slide:
  
    def __init__(self, file_path, show_thumbnail=False, verbose=False):

        self.file_path = file_path
        self.f_name = re.split("/", file_path)[-1].replace(".ndpi", "").strip()
        try:           
            self.slide = openslide.open_slide(self.file_path)
            self.base_file_name, ext = os.path.splitext(os.path.basename(file_path))       
            mpp_x = float(self.slide.properties[openslide.PROPERTY_NAME_MPP_X])
            mpp_y = float(self.slide.properties[openslide.PROPERTY_NAME_MPP_Y])
            
            self.dimensions = self.slide.dimensions
            self.avg_mpp = (mpp_x + mpp_y) / 2
            # physical width/height of slide in mm
            self.width = (self.dimensions[0] * mpp_x) / 1000
            self.height = (self.dimensions[1] * mpp_y) / 1000
       
            down_samples = self.slide.level_downsamples
            # top magnification although this may not be reliable
            self.objective_power = float(self.slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])     
            # calculate available magnifications; objective power value divided by each downsampling factor
            self.magnifications = tuple(self.objective_power / a for a in down_samples)
            if verbose:
                self.print_slide_details(show_thumbnail=show_thumbnail)
            
        
        except (openslide.OpenSlideError, FileNotFoundError) as e:
            self.slide = None
            raise e
          
  
    ##########################################################################################
    
    def get_tile_dimensions(self, magnification, tile_size_in_mm, tile_overlap_in_mm):
        """
        return a named tuple containing the coordinates and dimensions of tiles
        """
        # get info for this magnification
        level, factor = self.resolve_magnification(magnification)
        factors = self.slide.level_downsamples
        level_dims = self.slide.level_dimensions[level]
        
        
        # convert tissue in mm to pixels and overlap to pixels on WSI
        tile_size_in_pixels = self.convert_mm_to_pixels(tile_size_in_mm)        
        overlap = self.convert_mm_to_pixels(tile_overlap_in_mm) 
        # get corresponding sizes at the correct magnification
        tile_size = tile_size_in_pixels / factors[level]
        t_out = int(round(tile_size / factor))
        t_overlap = int(round((overlap / factors[level])/factor))
        
        # generate X, Y coordinates for tiling for read region (on original dimensions)
        x = np.arange(0, self.slide.dimensions[0], tile_size_in_pixels - overlap)
        y = np.arange(0, self.slide.dimensions[1], tile_size_in_pixels - overlap)   
        x, y = np.meshgrid(x, y)
        

        # get widths and heights of tiles at this magnification 
        x_plus_one = x + tile_size_in_pixels
        y_plus_one = y + tile_size_in_pixels
        np.clip(x_plus_one, 0, self.slide.dimensions[0], out=x_plus_one)
        np.clip(y_plus_one, 0, self.slide.dimensions[1], out=y_plus_one)
        width_unscaled, height_unscaled = x_plus_one - x, y_plus_one - y
        
        width = np.round((width_unscaled / factors[level])/factor).astype(int)
        height = np.round((height_unscaled / factors[level])/factor).astype(int)
        
        plan = collections.namedtuple('tile_plan', ['level', 't_out',
                                                    'factor', 
                                                    'magnification',  
                                                    'x', 'y', 
                                                    'width', 'height'])
        populate_plan = plan(level, t_out, factor, magnification, x, y, width, height)
        print(f"Magnification %.2f, %s, max %d tiles, %d x %d pixels, %d rows, %d columns" % (magnification, level_dims, (len(x[0]) * len(x)), t_out, t_out, len(x), len(x[0])))
        return populate_plan
    
    ##########################################################################################
        
    def get_valid_tile_mask(self, magnification, 
                            tile_size_in_mm, 
                            tile_overlap_in_mm,
                            tissue_percent_threshold, 
                            edge_percent_threshold, 
                            show_fg=False,
                            show_bg=False):
        
        """
        Choose a low res for speed here (magnification param)
        Returns a boolean 2d mask of good tiles (lots of tissue) and 
        bad tiles (too much background or edge slivers)
        
        magnification: choose from [40.0, 20.0, 10.0, 5.0, 2.5, 1.25, 0.625, 0.3125]
        tile_size_in_mm: how many mm of slide (square) per tile, 1.5 is about average
        tissue_percent_threshold: how much background is tolerable?
        edge_percent_threshold: minimum size a valid tile must be compared to a full square   
        If in a Jupyter notebook, can opt to display tiles as they are
        filtered in or out or both, with show_fg and show_bg
        """
        ts = self.get_tile_dimensions(magnification, tile_size_in_mm, tile_overlap_in_mm)
            
        tile_total = len(ts.x[0]) * len(ts.x)
    
        valid_mask = np.zeros((ts.x.shape[0], ts.x.shape[1]), dtype=bool)
           
        max_pixel_count = ts.t_out ** 2    

        for i in range(ts.x.shape[0]):
            for j in range(ts.x.shape[1]):
                x = int(ts.x[i, j])
                y = int(ts.y[i, j])
                dimension = (ts.width[i, j], ts.height[i, j])
                
                region = self.slide.read_region((x, y), 
                                                 ts.level, 
                                                 dimension)
                
                pixels = ts.width[i, j] * ts.height[i, j]
                
                complement = self.image_to_integer_greyscale_complement(region)      
                mask = self.hyst_threshold(complement, low=35, high=60)
                percent = (mask.sum() * 100) / pixels
               
                tile_ratio_to_max = int(pixels * 100 / max_pixel_count)
                
                # tile is kept if it has more foreground than the threshold
                # and also the tile isn't a tiny edge case
                if percent >= tissue_percent_threshold and tile_ratio_to_max >= edge_percent_threshold:                
                    
                    valid_mask[i][j] = True
                    if show_fg:
                        print("------ row %d, column %d" % (i, j))
                        print("Accept " + str(int(percent)) + "% tissue")
                        # for use in a Jupyter notebook setting only
                        display(region)
                else:
                    if show_bg:
                        # for use in a Jupyter notebook setting only
                        print("row %d, column %d" % (i, j))
                        print("Reject " + str(int(percent)) + "% tissue")
                        display(region)
        print("%d foreground tiles out of %d (%.1f percent threshold)" % (valid_mask.sum(), tile_total, tissue_percent_threshold))
        return valid_mask
    
    ##########################################################################################
    def get_valid_tile_mask_ilastik(self,background, magnification, 
                            tile_size_in_mm, 
                            tile_overlap_in_mm,
                            tissue_percent_threshold, 
                            edge_percent_threshold, 
                            show_fg=False,
                            show_bg=False):
        
        """
        Choose same res as background mask was created from (low res)
        Returns a boolean 2d mask of good tiles (lots of tissue) and 
        bad tiles (too much background or edge slivers)
        background: Background mask in PiL image format which is of same magnification as magnification specified here!
        magnification: choose from [40.0, 20.0, 10.0, 5.0, 2.5, 1.25, 0.625, 0.3125] # use same as to make masks
        tile_size_in_mm: how many mm of slide (square) per tile, 1.5 is about average
        tissue_percent_threshold: how much background is tolerable?
        edge_percent_threshold: minimum size a valid tile must be compared to a full square   
        If in a Jupyter notebook, can opt to display tiles as they are
        filtered in or out or both, with show_fg and show_bg
        """
        
        level, factor = self.resolve_magnification(magnification)
        factors = self.slide.level_downsamples
        
        # get tile grid
        ts = self.get_tile_dimensions(magnification, tile_size_in_mm, tile_overlap_in_mm)

        tile_total = len(ts.x[0]) * len(ts.x)
        
        # placeholder for valid tiles 
        valid_mask = np.zeros((ts.x.shape[0], ts.x.shape[1]), dtype=bool)
        red_mask = np.zeros((ts.x.shape[0], ts.x.shape[1]), dtype=float)
        
        # get the full expected tile size (for filtering smaller pieces out by edge percent threshold)
        max_pixel_count = ts.t_out ** 2    
        
        level_dims = self.slide.level_dimensions[ts.level]
        
        # get the scaled coords to search on the mask (which is generated at same magnification as used here)
        mask_x = np.round((ts.x / factors[level])/factor, 0).astype(int)
        mask_y = np.round((ts.y / factors[level])/factor, 0).astype(int)
        
        percent_red = []
        tile_number = 0
        # iterate through the tiles
        for i in range(ts.x.shape[0]):
            for j in range(ts.x.shape[1]):
                x = int(ts.x[i, j])
                y = int(ts.y[i, j])
                
                # get tile dimensions to check if edge case or not
                dimension = (ts.width[i, j], ts.height[i, j])
                pixels_in_tile = ts.width[i, j] * ts.height[i, j]
                tile_ratio_to_max = int(pixels_in_tile * 100 / max_pixel_count)
                 # keep if the tile isn't a tiny edge case
                if tile_ratio_to_max >= edge_percent_threshold:

                    # get mask for region and check tissue %
                    background_mask_region = background.crop((mask_x[i,j], mask_y[i,j], mask_x[i,j] + ts.width[i, j], mask_y[i,j] + ts.height[i, j]))
                    tissue = np.sum(np.array(background_mask_region))
                    pixels = background_mask_region.size[0]* background_mask_region.size[1]
                    percent = (tissue/pixels)*100
                    
                    region = self.slide.read_region((x, y), 
                                                     ts.level, 
                                                     dimension)
                    

                    # tile is kept if it has more foreground than the threshold
                    if percent >= tissue_percent_threshold:                
                        valid_mask[i][j] = True
                        
                        # get % red for tiles with enough tissue
                        red_percent = self.get_percent_red(region.convert("RGB"), lo=40, hi=200,show_tile=show_fg)
                        red_mask[i][j] = red_percent
                        
                        if show_fg:
                            print("------ row %d, column %d" % (i, j))
                            print("Accept " + str(int(percent)) + "% tissue")
                            f, (ax1, ax2) = plt.subplots(1,2)
                            ax1.imshow(region.convert("RGB"))
                            ax2.imshow(background_mask_region)
                            plt.show()
                    else:
                        if show_bg:
                            # for use in a Jupyter notebook setting only
                            print("row %d, column %d" % (i, j))
                            print("Reject " + str(int(percent)) + "% tissue")
                            f, (ax1, ax2) = plt.subplots(1,2)
                            ax1.imshow(region.convert("RGB"))
                            ax2.imshow(background_mask_region)
                            plt.show()
        
        return valid_mask , red_mask
    ##########################################################################################
    
    def get_percent_red(self, region, hi=200, lo=40, show_tile=False):
        # Threshold based on experimental values
        region = region.resize((224,224)) # to make easier to handle
        h,s,v = region.convert("HSV").split()
        res = np.where(((np.array(h) < lo) | (np.array(h) > hi)))
        region_array = np.zeros_like(h)
        region_array[res] = 1
        percent_red = 100* np.count_nonzero(region_array)/ region_array.size
        print("%d red pixels out of %d (%.1f percent red)" % (np.count_nonzero(region_array), region_array.size, percent_red))
        if show_tile:
            f, (ax1, ax2) = plt.subplots(1,2)
            ax1.imshow(region)
            ax2.imshow(region_array*255)
            plt.show()
        
        return percent_red
    ##########################################################################################
    
    
    def tile_and_save_jpg(self, output_dir, 
                          magnification, 
                          tile_size_in_mm, 
                          tile_overlap_in_mm,
                          jpg_quality, 
                          tile_mask=None):
        '''
        This is where the magic happens.
        Calculate tile dimensions, cut slide up, and save to JPEG
        
        output_dir: where to put, must exist
        magnification: choose from [40.0, 20.0, 10.0, 5.0, 2.5, 1.25, 0.625, 0.3125]
        tile_size_in_mm: how many mm of slide (square) per tile
        jpg_quality: compression rate for saving as JPEG; 95 is the max
        tile_mask: if provided, a 2d boolean array; must be of same dimension as the one generated here,
        make a prior call to get_valid_tile_mask() to create this. If none provided
        the output JPEGs will include background as well as foreground
        '''
        ts = self.get_tile_dimensions(magnification, tile_size_in_mm, tile_overlap_in_mm)
        
        max_tiles = ts.x.shape[0] * ts.x.shape[1]
        valid_tiles = max_tiles
        
        if tile_mask is not None:        
            if tile_mask.shape != ts.x.shape:
                problem = "Mask {0} and tile pattern {1} are different dimensions".format(tile_mask.shape, ts.x.shape)
                raise ValueError(problem)    
         
            valid_tiles = tile_mask.sum()  

        output_dir = os.path.join(output_dir, "_".join([self.base_file_name, 
                               "tiles", 
                               "x" + str(magnification), 
                               str(tile_size_in_mm) + "mm", 
                               "q" + str(jpg_quality)]))
        
        
        os.makedirs(output_dir, exist_ok=True)
        
        
        for i in range(ts.x.shape[0]):
            for j in range(ts.x.shape[1]):               
                x = int(ts.x[i, j])
                y = int(ts.y[i, j])
                dimension = (ts.width[i, j], ts.height[i, j])
                
                img_name = "_".join([ "tile", str(i), str(j) ]) + ".jpg"
                img_save_path = os.path.join(output_dir, img_name)   
                 
                # if a validity mask has been passed in then use it 
                # to discriminate, otherwise save all             
                if tile_mask is None:
                    self.save_image_from_region(ts.level, 
                                                x, y, 
                                                dimension, 
                                                img_save_path, 
                                                jpg_quality)  
                elif tile_mask[i][j]:             
                    self.save_image_from_region(ts.level, 
                                                x, y,
                                                dimension, 
                                                img_save_path, 
                                                jpg_quality)
                    
        stats = {}
        
        stats["ndpi_file"] = self.file_path        
        stats["ndpi_file_size_mb"] = "{:.1f}".format(os.path.getsize(self.file_path) / (1024 * 1024))
        stats["high_res"] = self.dimensions
        stats["tile_magnification"] = magnification
        stats["tile_size_mm"] = tile_size_in_mm
        stats["tile_size_pixels"] = ts.t_out
        stats["max_tiles"] = max_tiles
        stats["mostly_tissue"] = valid_tiles
        stats["mostly_background"] = max_tiles - valid_tiles
        stats["output_dir"] = output_dir
        
        return stats
    
    
    
     ##########################################################################################
    
    def tile_and_save_jpg_ilastik(self, 
                          magnification, 
                          tile_size_in_mm, 
                          tile_overlap_in_mm, 
                          tile_mask=None, red_mask=None, show_tile=False):
        '''
        This is where the magic happens.
        Calculate tile dimensions, cut slide up, and save coords to CSV
        magnification: choose from [40.0, 20.0, 10.0, 5.0, 2.5, 1.25, 0.625, 0.3125]
        tile_size_in_mm: how many mm of slide (square) per tile
        tile_mask: if provided, a 2d boolean array; must be of same dimension as the one generated here,
        make a prior call to get_valid_tile_mask() to create this. If none provided
        the output list of tiles will include background as well as foreground
        
        Saves the coords of valid tiles (required in read_region / save_image_from_region()) to stats["tile_info"]
        returns: dictionary with tiling info

        '''
        ts = self.get_tile_dimensions(magnification, tile_size_in_mm, tile_overlap_in_mm)
        factors = self.slide.level_downsamples
        level, factor = self.resolve_magnification(magnification)
        max_tiles = ts.x.shape[0] * ts.x.shape[1]
        valid_tiles = max_tiles
        
        tile_info = []
        red_list = []
        
        if tile_mask is not None:        
            if tile_mask.shape != ts.x.shape:
                problem = "Mask {0} and tile pattern {1} are different dimensions".format(tile_mask.shape, ts.x.shape)
                raise ValueError(problem)    
            valid_tiles = tile_mask.sum() 
        
        if red_mask is not None:
            if red_mask.shape != ts.x.shape:
                problem = "Red Mask {0} and tile pattern {1} are different dimensions".format(red_mask.shape, ts.x.shape)
                raise ValueError(problem) 

        
        for i in range(ts.x.shape[0]):
            for j in range(ts.x.shape[1]):               
                x = int(ts.x[i, j])
                y = int(ts.y[i, j])
                dimension = (ts.width[i, j], ts.height[i, j])
                
                
                # if a validity mask has been passed in then use it 
                # to discriminate, otherwise save all             
                if tile_mask is None:
                    tile_info.append(f"{int(ts.level)}/{x}/{y}/{dimension[0]}/{dimension[1]}")
                    
                elif tile_mask[i][j]: 
                    if show_tile:
                        
                        print(f"{ts.level}/{x}/{y}/{dimension[0]}/{dimension[1]}/{red_mask[i][j]}")
                        region = self.slide.read_region((x, y), 
                                                     ts.level, 
                                                     dimension).convert("RGB")
                        plt.figure()
                        plt.imshow(region)
                        plt.title(f"Image size: {region.size}")
                        plt.show()
                        
                    
                    
                    tile_info.append(f"{int(ts.level)}/{x}/{y}/{dimension[0]}/{dimension[1]}/{red_mask[i][j]}")
                    if red_mask is not None:
#                         print(f"This tile has {red_mask[i][j]} percent red pixels")
                        red_list.append(red_mask[i][j])
                    
        stats = {}
        
        stats["ndpi_file"] = self.file_path        
        stats["ndpi_file_size_mb"] = "{:.1f}".format(os.path.getsize(self.file_path) / (1024 * 1024))
        stats["high_res"] = self.dimensions
        stats["tile_magnification"] = magnification
        stats["tile_size_mm"] = tile_size_in_mm
        stats["tile_size_pixels"] = ts.t_out
        stats["max_tiles"] = max_tiles
        stats["mostly_tissue"] = valid_tiles
        stats["mostly_background"] = max_tiles - valid_tiles
        stats["tile_info"] = tile_info
        stats["red_info"] = red_list
        return stats
    
    ##########################################################################################
          
    def save_image_from_region(self, level, 
                               x, y, 
                               dimension, 
                               out_file_path, 
                               jpg_quality):
        """
        Convert slide region to RGB PIL image and save as JPEG
        level: the OpenSlide 'level' which is the zoom factor
        x, y, dimension: defines the region
        (see https://openslide.org/api/python/)
        out_file_path: where to put, must exist
        jpg_quality: compression rate for saving as JPEG; 95 is the max
        """
        region = self.slide.read_region((x, y), level, dimension)
        img = region.convert("RGB")      
        img.save(out_file_path, format="JPEG", quality=jpg_quality)
    
    ##########################################################################################
        
    def save_as_jpg(self, output_dir, magnification, jpg_quality):
        '''
        Whole slide saved to JPEG at given magnification and quality.
        
        output_dir: where to put, must exist
        magnification: choose from [40.0, 20.0, 10.0, 5.0, 2.5, 1.25, 0.625, 0.3125]
        jpg_quality: compression rate for saving as JPEG; 95 is the max
        '''    

        level, scale = self.resolve_magnification(magnification)
        img_name = self.base_file_name + ".jpg"
        img_save_path = os.path.join(output_dir, img_name)
        self.save_image_from_region(level, 
                                    0, 0, 
                                    self.slide.level_dimensions[level], 
                                    img_save_path, 
                                    jpg_quality)
            
    
    ##########################################################################################
    
    
    
    def resolve_magnification(self, mag, tol=0.002):
    
        '''
        Reverse slide 'level' and scale factor from magnification value for convenience
        tol: tolerance between available magnifications and specified
        '''  
        try:
            assert mag in self.magnifications, f"{mag} must be in {self.magnifications}"
        except AssertionError:
            raise AssertionError(f"{mag} must be in {self.magnifications}")
        
        mismatch = tuple(x - mag for x in self.magnifications)
      
        abs_mismatch = tuple(abs(x) for x in mismatch)
        
        if min(abs_mismatch) <= tol:
            level = int(abs_mismatch.index(min(abs_mismatch)))
            factor = 1
       
        return level, factor
    

    '''
    Filter techniques below gratefully borrowed from https://github.com/CODAIT/deep-histopath
    '''

    def np_to_pil(self, np_img):
    
        if np_img.dtype == "bool":
            np_img = np_img.astype("uint8") * 255
        elif np_img.dtype == "float64":
            np_img = (np_img * 255).astype("uint8")
        return PIL.Image.fromarray(np_img)
    
    def filter_threshold(self, np_img, threshold, output_type="bool"):
    
        result = (np_img > threshold)
        if output_type == "bool":
            pass
        elif output_type == "float":
            result = result.astype(float)
        else:
            result = result.astype("uint8") * 255
        return result
    
    def hyst_threshold(self, np_img, low=35, high=60, output_type="bool"):
        
        result = sk_filters.apply_hysteresis_threshold(np_img, low, high)
        
        if output_type == "bool":
            pass
        elif output_type == "float":
            result = result.astype(float)
        else:
            result = result.astype("uint8") * 255
        return result
    
    def filter_complement(self, img_as_array, output_type="uint8"):
    
        if output_type == "float":
            complement = 1.0 - img_as_array
        else:
            complement = 255 - img_as_array
        return complement
    
    def image_to_integer_greyscale_complement(self, img):
        
        try:    
            grey_image = img.convert("L")
            img_as_array = np.asarray(grey_image) 
            complement = self.filter_complement(img_as_array)
        except TypeError as e:           
            print(e)
            complement = None
        return complement
    
    ##########################################################################################
    
    def convert_mm_to_pixels(self, width_in_mm):
        '''
            Calculate the number of pixels for a given physical width in mm
            tile_size_in_mm: a value in mm for each tiles's width and height
        ''' 
        return int(width_in_mm * 1000 /self.avg_mpp)
    
    def convert_pixels_to_mm(self, magnification, pixels):
        # takes in pixels measured on image of magnificaiton given
        # converts to mm of tissue on actual original WSI
        level, factor = self.resolve_magnification(magnification)
        factors = self.slide.level_downsamples
        lo_mag_thickness = pixels * factor * factors[level]
        tissue_thickeness = lo_mag_thickness* self.avg_mpp/1000 
        return tissue_thickeness
   ##########################################################################################
    
    def print_slide_details(self, show_thumbnail=False, 
                            max_size=(600, 400)):
        print("-" * 100)
        print("File:", os.path.basename(self.file_path))                 
        print("Dimensions", self.slide.dimensions)
        print("File size in MB %.1f" % (os.path.getsize(self.file_path) / (1024 * 1024)))
        #print(f"Microns per pixel %.3f" % (self.avg_mpp))
        print(f"Physical width %.2f mm" % self.width)
        print(f"Physical height %.2f mm" % self.height)        
        #print("Magnifications", self.magnifications)
        
        if show_thumbnail:
            # for use in a Jupyter notebook setting only
            display(self.slide.get_thumbnail(size=max_size))
        print("-" * 100)

    ##########################################################################################

    def close(self):
        if self.slide is not None:
            self.slide.close()
            
    ##########################################################################################

    


    