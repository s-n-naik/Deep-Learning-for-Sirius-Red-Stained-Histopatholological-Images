# CHANGE IF ON DIFF COMPUTER
import glob
import os
import pandas as pd
import numpy as np
import cv2
from ndpi_slide import Slide
import matplotlib.pyplot as plt
import os
import shutil
import uuid
import h5py
from PIL import Image
from joblib import Parallel, delayed
import argparse
import time
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.color import label2rgb
from skimage import img_as_bool
from scipy.ndimage.measurements import label
from skimage.measure import label, regionprops
import math



my_parser = argparse.ArgumentParser(description='create_patches')

my_parser.add_argument('--score_xls', type=str, default='/data/MSc_students_accounts/sneha/sneha/sirius_red-master/score_file_matches.csv', help='path to the scores_file_matches.csv file')
my_parser.add_argument('--base_dir', type=str, default='/data/MSc_students_accounts/sneha/', help='path to where low res jpg and masks will be saved')
my_parser.add_argument('--data_dir', type=str, default='/data/goldin/images_raw/nafld_liver_biopsies_sirius_red/', help='path to where ndpi raw data is saved')
my_parser.add_argument('--lo_mag', type=float, default=1.25 , help='magnification for masks + tiling choose from [40.0, 20.0, 10.0, 5.0, 2.5, 1.25, 0.625, 0.3125]')
my_parser.add_argument('--hi_mag', type=float, default=5 , help='magnification for viewing final tiles choose from [40.0, 20.0, 10.0, 5.0, 2.5, 1.25, 0.625, 0.3125]')

my_parser.add_argument('--tissue_width_mm', type=float, default=2.5 , help='length of slide to include in a single tile, in mm')
my_parser.add_argument('--tile_overlap', type=int, default=50 , help='length of slide to overlap, in % of tile size')
my_parser.add_argument('--tissue_percent_threshold', type=int, default=50 , help='percentage of tissue in tile must be greater than this threshold int between 0 and 100')
my_parser.add_argument('--edge_percent_threshold', type=int, default=100, help='a tile must be at least this percent size of the full square ones to be considered acceptable (takes out the small edge pieces) between 0 and 100')
my_parser.add_argument('--lo_jpg_quality', type=int, default=50, help='jpg quality to save down the reference WSI images (95 max)')
my_parser.add_argument('--adapt_method', type=str, default="threshold", help='If there are too few tiles in a bag, this method (patch_size or threshold) will be used to adjust parameters and increase tiles per bag')

my_parser.add_argument('--create_tiles', type=str, default="y",  help='select tiles and generate CSV with location info (y/n)')
my_parser.add_argument('--mask_dir', type=str, default="n",  help='if provided, existing masks will be used rather than creating masks')
my_parser.add_argument('--jpg_dir', type=str, default="n",  help='if provided, existing jpgs will be used rather than creating jpgs')

my_parser.add_argument('--ilastik_dir', type=str, default="/data/MSc_students_accounts/sneha/ilastik-1.3.3post3-Linux",  help='path to where the ilastik repo is located')
args = my_parser.parse_args()

# e.g. python3 create_patches.py --base_dir="/data/MSc_students_accounts/sneha/" --mask_dir="/data/MSc_students_accounts/sneha/wsi_mask_ca594bc0-6d05-4dc3-bdf8-10ec49ce7326" --create_tiles="y" --hi_mag=10 --tissue_width_mm=2.5 --tile_overlap=1



# Patching Class
class Patching:
    def __init__(self, 
                 score_xls_path: str, 
                 base_dir: str,
                 data_dir: str,
                 lo_mag: float = 1.25,
                 hi_mag: float = 4,
                 tissue_width_mm: float = 2.5,
                 tile_overlap_percent: int = 0,
                 tissue_percent_threshold: int = 20, 
                 edge_percent_threshold: int = 100,
                 lo_jpg_quality:int = 50, mask_dir="n", jpg_dir="n", ilastik_dir = "/data/MSc_students_accounts/sneha/ilastik-1.3.3post3-Linux",adapt_method=None):
        
        self.string_list = None
        self.all_files = None
        
        self.base_dir = base_dir
        self.data_dir = data_dir
        self.ilastik_path =  os.path.abspath(ilastik_dir)
        self.adapt_method = adapt_method
        self.lo_mag = lo_mag
        self.hi_mag = hi_mag
        self.tissue_width_mm = tissue_width_mm
        self.tile_overlap = tile_overlap_percent/100
        self.tissue_percent_threshold = tissue_percent_threshold
        self.edge_percent_threshold = edge_percent_threshold
        self.lo_jpg_quality = lo_jpg_quality
        
        self.score_xls = self._get_score_xls(score_xls_path)
        self._create_directories(mask_dir, jpg_dir)
        
        

    def _get_score_xls(self, score_xls_path):
        # read files from matched list in the CSV file, with scores
        score_xls = pd.read_csv(os.path.abspath(score_xls_path))
        print(f'Scores xls size before removal: {len(score_xls)}')
        # manual scrubbing of a few problem files TODO: make it an input?
        score_xls = score_xls[~score_xls['biopsy_id_clean'].isin(['09-27536','11-8490','12-10928' , '12-677','12-8034','13-15702','13-9454','14-2382','15-14307','15-18963','15-8853','15-9741','15-19588', '16-2206', '16-8332','16-8340'])]
        print(f'biopsy id clena: {len(score_xls)}')
        # only keep the rows where we have slides
        missing_slide = score_xls["biopsy_id_matched"].isnull()
        score_xls = score_xls[~missing_slide].drop_duplicates(subset='biopsy_id_matched')#.dropna() # Element '15-11971' is present twice
        print(f'Scores xls size after removal: {len(score_xls)}')
        score_xls.dropna(subset=['biopsy_id_matched'])
        print(f'Scores xls size after removal: {len(score_xls)}')
        # check data_dir is correct
        score_xls["biopsy_id_matched"] = score_xls["biopsy_id_matched"].astype(str).apply(lambda x: "".join([self.data_dir, x.split("/")[-1]]))
        files = score_xls["biopsy_id_matched"]
        # get all the file paths we will process and save them
        files_in_data_dir = list(glob.glob(f"{self.data_dir}*.ndpi"))
        # intersection of the files we have in score_file_matches and those we have in data_dir i.e. ones we can actually process
        self.all_files = list(set(files_in_data_dir).intersection(set(files)))
        print(f"There are {len(self.all_files)} files to process")
        return score_xls

    def _create_directories(self, mask_dir, jpg_dir):
        # directory for small reference jpgs
        if jpg_dir != "n":
            self.jpg_dir = os.path.abspath(jpg_dir)
            print(f"Using jpgs in {self.jpg_dir}")
        else:
            jpg_dir_new = os.path.join(self.base_dir, 
                                      "_".join([ "reference_jpgs", 
                                               str(uuid.uuid4())]))
            os.makedirs(jpg_dir_new, exist_ok = True)
            self.jpg_dir = jpg_dir_new
            print(f"Created directory for jpgs \n {self.jpg_dir}")
            
        if mask_dir != "n":
            self.mask_dir = os.path.abspath(mask_dir)
            print(f"Using masks in {self.mask_dir}")
        else:
            # directory where .h5 masks will be placed
            mask_dir_new = os.path.join(self.base_dir, "_".join([ "wsi_mask", str(uuid.uuid4())]))
            os.makedirs(mask_dir_new, exist_ok = True)
            self.mask_dir = mask_dir_new
            print(f"Created directory for masks \n{self.mask_dir}")



    def create_low_res_images(self, file):
        slide = Slide(file, False)
        # save a lo-res whole-slide JPG for quick reference
        slide.save_as_jpg(self.jpg_dir, self.lo_mag, self.lo_jpg_quality)

    def run_create_ref_image(self):
        print("Starting to create reference jpg images")
        Parallel(n_jobs=-1, verbose=5)(delayed(self.create_low_res_images)(i) for i in self.all_files)
        print(f"Created ref images for {len(glob.glob(f'{self.jpg_dir}/*.jpg'))} files")
    
    @staticmethod
    def group_files_in_batches(all_files, grouping: int = 5):
        assert len(all_files) > 0, "There are no files to group"
        num_groups = len(all_files)// grouping
        left_over = len(all_files)% grouping
        string_list = [(i*grouping, min((i+1)*grouping, len(all_files))) for i in range(num_groups+1)]
        string_list = [" ".join(all_files[i:j]) for i,j in string_list]
        string_list = list(filter(None, string_list))
        num_groups = len(all_files)// grouping
        string_list = [" ".join(all_files[i:i+grouping]) for i in range(num_groups)]
        print(f"We have {len(string_list)} batches of files")
        return string_list



    def run_ilastik(self,file_string):
        os.chdir(os.path.abspath(self.ilastik_path))
        os.system(f"./run_ilastik.sh --headless --project=Ilastik_pixel_segmentation.ilp  {file_string}")

    def run_create_masks_ilastik(self):
        starttime = time.time()
        
        os.chdir(os.path.abspath(self.ilastik_path))
        all_files = list(glob.glob(f"{self.jpg_dir}/*.jpg"))
        print(f"Creating reference masks for {len(all_files)} ref jpgs")
        Parallel(n_jobs=-1, verbose=10)(delayed(self.run_ilastik)(i) for i in all_files)
        list_of_masks = glob.glob(f"{self.jpg_dir}/*Probabilities.h5")
        print(f"Created {len(list_of_masks)} in jpg dir {self.jpg_dir} in {time.time() - starttime}s")
        print(f"Moving these to {self.mask_dir}")
        for mask_path in list_of_masks:
            shutil.move(os.path.abspath(mask_path), os.path.abspath(self.mask_dir))
        print(f"There are now {len(os.listdir(os.path.abspath(self.mask_dir)))} masks in {self.mask_dir}")
         
 
        
    def get_background_mask(self,mask_path):
        try:
            mask = h5py.File(os.path.abspath(mask_path))
        except:
            return None
        mask = np.array(mask["exported_data"]).transpose(2,0,1)
        label_1 , label_2 = mask[0], mask[1]
        background_mask = label_1 > label_2
        return Image.fromarray(background_mask)


    def select_tiles(self,f, verbose=False):
        '''
        NOTE: lo_mag here MUST be the same as lo_mag used to create the masks
        Loads the WSI slide and corresponding background mask
        Selects valid tiles and gets an array of bools for whether tiles in the WSI have enough tissue (> threshold)
        Goes over tile at hi magnification and gets the corresponding regions from the tile that are valid and saves down in a dict
        '''
        stats_dict = {}
        slide = Slide(os.path.abspath(f), show_thumbnail=True, verbose=verbose) 
        starttime= time.time()
        # get background mask
        background_mask_path = os.path.join(self.mask_dir, slide.f_name + "_Probabilities.h5")
        background = self.get_background_mask(os.path.abspath(background_mask_path))
        if background is None:
            print(f"No background mask for {f}")
            slide.close()
            return stats_dict
        else:
            # 3 key variables
            tile_overlap_mm = round(self.tile_overlap*self.tissue_width_mm,1)
            tissue_width_mm = self.tissue_width_mm
            tissue_threshold = self.tissue_percent_threshold
            
            starttime = time.time()
            mask, red_mask = slide.get_valid_tile_mask_ilastik(background, self.lo_mag, 
                                             tissue_width_mm, 
                                             tile_overlap_in_mm=tile_overlap_mm,
                                             tissue_percent_threshold=tissue_threshold, 
                                             edge_percent_threshold=self.edge_percent_threshold,
                                             show_fg=verbose, 
                                             show_bg=verbose) # set to True for Jupyter only
            
            # if number of valid tiles < 10, check the tissue size and reduce patch size, otherwise reduce tissue % threshold
            if np.sum(mask) < 10:
                print(f"There are less than 10 valid tiles at {self.tissue_width_mm} for slide {f}")
                if self.adapt_method == "patch_size":
                    
                    # update 3 key variables
                    tissue_width_mm = self.get_tissue_thickness_convex_hull(background, slide, verbose=False)
                    tile_overlap_mm = round(self.tile_overlap*tissue_width_mm,1)
                    tissue_threshold = self.tissue_percent_threshold
                    print(f"Adjusting tissue width for tiles to {tissue_width_mm} and overlap to {tile_overlap_mm} and re-running")
                    
                    mask, red_mask = slide.get_valid_tile_mask_ilastik(background, self.lo_mag, 
                                             tissue_width_mm, 
                                             tile_overlap_in_mm=tile_overlap_mm,
                                             tissue_percent_threshold=tissue_threshold, 
                                             edge_percent_threshold=self.edge_percent_threshold,
                                             show_fg=verbose, 
                                             show_bg=verbose)
                elif self.adapt_method == "threshold":
                    #update 3 key variables
                    tile_overlap_mm = round(self.tile_overlap*self.tissue_width_mm,1)
                    tissue_width_mm = self.tissue_width_mm
                    tissue_threshold = 0.5*self.tissue_percent_threshold
                    print(f"The current tissue % threshold is {self.tissue_percent_threshold}, Adjusting this to {tissue_threshold} and re-running")
                    mask, red_mask = slide.get_valid_tile_mask_ilastik(background, self.lo_mag, 
                                             tissue_width_mm, 
                                             tile_overlap_in_mm=tile_overlap_mm,
                                             tissue_percent_threshold=tissue_threshold, 
                                             edge_percent_threshold=self.edge_percent_threshold,
                                             show_fg=verbose, 
                                             show_bg=verbose) 
                    

            print(f"Tile mask created in {time.time() - starttime} ")

            starttime = time.time()

            stats_dict = slide.tile_and_save_jpg_ilastik(self.hi_mag,
                                                 tissue_width_mm,
                                                 tile_overlap_mm, mask, red_mask, show_tile=True)

            # optionally add the scores to the summary output
            if self.score_xls is not None:
                score_xls = self.score_xls
                full_path = str(f)
                try:
                    stats_dict["total_score"] = score_xls[score_xls["biopsy_id_matched"]==full_path]['total_score'].iloc[0]
                    stats_dict["stage"] = score_xls[score_xls["biopsy_id_matched"]==full_path]['stage'].iloc[0]
                except:
                    stats_dict["total_score"] = None
                    stats_dict["stage"] = None
                try:
                    stats_dict["color_characteristics"] = score_xls[score_xls["biopsy_id_matched"]==full_path]['color_characteristics'].iloc[0]
                except:
                    stats_dict["color_characteristics"] = None
                finally:
                    stats_dict["tissue_threshold"] = tissue_threshold
                    stats_dict["tile_overlap"] = tile_overlap_mm
                    stats_dict["tissue_width"]= tissue_width_mm


            print(f"Tiles gathered in {time.time() - starttime}")

            slide.close()
            return stats_dict 
                           
    def run_create_tiles(self, verbose=False):
        # changing into data directory
        os.chdir(os.path.abspath(self.data_dir))
        # initialise dataframe for summary info
        df = pd.DataFrame(columns = ["ndpi_file", "ndpi_file_size_mb", "high_res", "tile_magnification", 
                                     "tile_size_mm", "tile_size_pixels", "max_tiles", 
                                     "mostly_tissue", "mostly_background","stage", "total_score"])
        stats_dict = {}
        stats_dict = Parallel(n_jobs=-1, verbose=10)(delayed(self.select_tiles)(i,verbose) for i in self.all_files)
                           
                           
        return stats_dict
    

    def save_stats_dict_to_file(self, f_name, stats_dict):
        my_df = pd.DataFrame(stats_dict)
        my_df = my_df.dropna(subset=["ndpi_file"])
        max_tiles = int(my_df.loc[my_df["mostly_tissue"].argmax()]["mostly_tissue"])
        columns = list(my_df.columns) + list(np.arange(max_tiles))
        final_df = pd.DataFrame(columns=columns)
        for i in range(len(my_df)):
            row = my_df.iloc[i]
            if row["mostly_tissue"] == 0:
                print("No tiles for WSI", row["ndpi_file"])
                slide = Slide(os.path.abspath(row["ndpi_file"]), show_thumbnail=True, verbose=True)
                slide.close()
                continue
            tile_info = row["tile_info"]
            new_row = pd.DataFrame(tile_info)
            concat_row = pd.concat([row ,new_row], ignore_index=False, axis=0).T
            final_df = pd.concat([final_df, concat_row], axis=0, ignore_index=True)
        final_df.to_csv(os.path.join(self.base_dir, f_name))

        print(f"Stats dict saved in {os.path.join(self.base_dir, f_name)}")
        
        return os.path.join(self.base_dir, f_name)
    
    
    def get_tissue_thickness(self, background, slide, verbose=False):
        folder_path = os.path.abspath(os.path.join(self.base_dir,"background_jpg"))
        os.makedirs(folder_path, exist_ok=True)
        uuid_name = str(uuid.uuid4())
        f_name = f"background_test_{uuid_name}.jpg"
        f_path = os.path.join(folder_path, f_name)
        background.save(f_path)
        background = cv2.imread(f_path, 0)
        background_rgb = cv2.imread(f_path)
        # using findContours func to find the none-zero pieces
        contours, hierarchy = cv2.findContours(background,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        thickeness_list = []
        for i in range(len(contours)):
            # get bounding box from contour
            minRect = cv2.minAreaRect(contours[i])
            ((x, y), (width, height), angle_of_rotation)  = minRect
            tissue_width_pixels = min(width, height)
            thickness_mm = slide.convert_pixels_to_mm(lo_mag, tissue_width_pixels)
            if (thickness_mm > 0.5 and thickness_mm < 5):
                thickeness_list.append(thickness_mm)
                box = cv2.boxPoints(minRect)
                box = np.intp(box) #np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
                cv2.drawContours(background_rgb, [box], 0, (0,255,0), -1)
        if verbose:
            print(f"Found {len(thickeness_list)} rectangles with widths: {thickeness_list}")
            print(f"Mean: {np.mean(thickeness_list)} +/- {np.std(thickeness_list)}")
            f, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))
            ax1.imshow(background_rgb)
            ax2.imshow(background)
            plt.show()
            slide.close()
            
        
        # delete file path
        os.remove(os.path.abspath(f_path)) 
        return np.mean(thickeness_list), np.std(thickeness_list)
        
    def get_tissue_thickness_convex_hull(self, background, slide, verbose=False, hole_size=3000, min_area=10000, floor=1, ceil=2.5):
        background = img_as_bool(background)
        background = remove_small_objects(background, hole_size)
        background = remove_small_holes(background, hole_size)
        label_image = label(background)
        image_label_overlay = label2rgb(label_image, image=background, bg_label=0)
        if verbose:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(image_label_overlay)
        valid_tissue_thickness = []
        for props in regionprops(label_image):
            '''
            0.5mm = 70
            1mm = 140
            1.5mm = 210
            2mm = 280
            2.5mm = 345
            '''
            if props.area > min_area:
                minor = round(props.major_axis_length,1)
                major = round(props.minor_axis_length,1)
                if verbose:
                    print(f"Major/ Minor / Area: {major}/{minor}/ {props.area}")
                if 50 < min(major, minor):
                    valid_tissue_thickness.append(min(major, minor))
                    if verbose:
                        y0, x0 = props.centroid
                        orientation = props.orientation
                        x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
                        y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
                        x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
                        y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length
                        ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
                        ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
                        ax.plot(x0, y0, '.g', markersize=15)

                        minr, minc, maxr, maxc = props.bbox
                        bx = (minc, maxc, maxc, minc, minc)
                        by = (minr, minr, maxr, maxr, minr)
                        ax.plot(bx, by, '-b', linewidth=2.5)

        if verbose:
            ax.set_axis_off()
            plt.tight_layout()
            plt.show()
        
        mean_px, std = np.mean(valid_tissue_thickness), np.std(valid_tissue_thickness)
        
        valid_mm = slide.convert_pixels_to_mm(self.lo_mag,mean_px)
        
        new_tissue_size = round(max(min(valid_mm, ceil),floor),1)
        
        if verbose:
            print(f"Valid tissue {round(mean_px)}px/{round(valid_mm,1)}")
        print(f"New tissue size: {new_tissue_size}")
        
        return new_tissue_size

        

    


# In[ ]:

def main(args):
    print("Initialising patching")
    print(args)
    overallstarttime = time.time()
    patching = Patching(score_xls_path = args.score_xls, 
                 base_dir = args.base_dir,
                 data_dir = args.data_dir,
                 lo_mag = args.lo_mag,
                 hi_mag = args.hi_mag,
                 tissue_width_mm=args.tissue_width_mm,
                 tile_overlap_percent=args.tile_overlap,
                 tissue_percent_threshold=args.tissue_percent_threshold, 
                 edge_percent_threshold=args.edge_percent_threshold,
                 lo_jpg_quality=args.lo_jpg_quality,
                mask_dir=args.mask_dir, jpg_dir=args.jpg_dir, ilastik_dir=args.ilastik_dir, adapt_method=args.adapt_method)
    
    print("Jpg dir provided", args.jpg_dir)
    starttime = time.time()
    if args.jpg_dir == "n":
        print("No jpg dir provided, creating new jpg ref images")
        patching.run_create_ref_image()
        print(f"Created {len(os.listdir(os.path.abspath(patching.jpg_dir)))} ref images in {time.time() - starttime}")
    else:
        print(f"Using jpgs provided in {args.jpg_dir}")
        
        
    print("Mask dir provided", args.mask_dir)
    starttime = time.time()
    if args.mask_dir == "n":
        print("No mask dir provided, creating masks with ilastik")
        patching.run_create_masks_ilastik()
        print(f"Created {len(os.listdir(os.path.abspath(patching.mask_dir)))} masks in {time.time() - starttime}")
    else:
        print(f"Using masks provided in {args.mask_dir}")
    
    
    
    print("Creating tiles and saving down tiles_summary.csv", args.create_tiles)
    starttime = time.time()
    if args.create_tiles == "y":
        stats_dict = patching.run_create_tiles()
        print(f"Tiled {len(stats_dict)} WSIs in {time.time() - starttime}")
        results_path = patching.save_stats_dict_to_file(f"tiles_summary_{args.hi_mag}x_{args.tissue_width_mm}mm_{args.tile_overlap}%_adaptive_{args.adapt_method}.csv", stats_dict)
        print(f"Finished in {time.time() - overallstarttime}! Tiling results can be found in {results_path}")
        
        
        
        
        

if __name__ == '__main__':
    main(args)

