# Deep Learning for Stratifying Fibrosis on Histopathological Images of the Liver

This repository contains the implementation to pre-process raw Sirius-Red stained, digitized liver biopsy samples (in partnership with St Mary's Paddington Hospital) for the diagnosis of Non-alcoholic Fatty Liver Disease. One can then train a Deep Learning model for a binary classifcation task of Severe vs Mild fibrosis using a Multiple Instance Learning framework with or without the use of augmentation, stain normalisation or unsupervised pretraining and finally we generate patch-level attention visualisations of training 'bags'.

## Set-up

- Python >= 3.8.8, NVIDIA GPU 

- Clone the repository to the desired location
`` git clone https://gitlab.doc.ic.ac.uk/malpatliv/sneha.git ``

- Download and install anaconda / miniconda onto your machine: https://www.anaconda.com/distribution/

- Create and activate a new virtual environment using conda or pip

- cd into root of this repository and run `sh install_packages.sh`
    - Installs dependencies including PyTorch, numpy, openslide-python, h5py, matplotlib, opencv-python, scikit-learn, scipy, pillow, pandas, pytorch
    - Finally it will install the Ilastik Pixel workflow [1] and copy the pre-trained model 'Ilastik_pixel_segmentation.ilp' into the Ilastik directory
## Repository Guide

### WSI Patching
To generate a CSV file for training our models, run `python3 sirius-red_master/create_patches.py` with the following command line arguments:
- --score_xls: path to the csv file with WSI (.ndpi) filenames and corresponding fibrosis labels
- --base_dir: path where the low resolution jpgs and background masks will be saved
- --data_dir: path to the folder containing the raw data
- --ilastik_dir: path to where the Ilastik repo and trained model are located (root of this repository)

Additional parameters that can be passed (defaults) include magnification (lo = 1.25x, hi = 5x), tissue size mm (2.5), tile overlap % (50), edge % threshold (100), tissue % threshold (50), adapt method (threshold). If masks or reference jpgs have already been generated, the locations of these can be passed using --jpg_dir and --mask_dir arguments to speed up the patching process.

    ├── base_dir/
        ├── jpg_dir/
            ├── slide_1.jpg
            ├── slide_2.jpg
            └── ...
        ├── mask_dir/
            ├── slide_1.h5
            ├── slide_2.h5
            └── ...

        ├── tiles_summary_{hi_mag}x_{tissue_width_mm}mm_{tile_overlap}%_adaptive_{adapt_method}.csv
	└── ...


create_patches.py will use ilastik to segment every WSI in data_dir, and save down reference jpgs and background masks in the directories above. It will then extract all patches with tissue content greater than 'tissue % threshold' and save the locations of these patches to a csv file output tiles_summary_{hi_mag}x_{tissue_width_mm}mm_{tile_overlap}%_adaptive_{adapt_method}.csv. An example patching csv file is uploaded.

### Experiments

- Experiments use the Weights and Biases [2] software for logging metrics and visualisations. Hyperparameters can be adjusted using the configuration dictionaries under  `if __name__ == '__main__':`. The tile_path parameter should be adjusted to contain the absolute path to the output of the WSI patching


    - run 'run_inference.py' to load pre-trained model weights and run inference on a dataset tiled using create_patches.py. Accuracy, F1 score and total confusion matrix are returned
    - run `run_finetune.py` to conduct supervised finetuning of models
        - training parameters that can be configured: num_epochs, batch_size, learning rate, learning rate scheduler, test and validation folds, whether to use early stopping
        - dataloading parameters that can be configured: dataloading method (multiple inference or single bags per WSI), bag size
        - architectural parameters that can be configured: use_ssc (if True then can modify SSC architecture (M,N,R,S) and reconstruction loss), , MIL architecture (max pooling, gated attention), Feature extractor (resnet18, se_resnet18, resnet34, se_resnet34, simclr pretrained), weight intialisation from load path.

    - run `unsupervised_training.py` to conduct unsupervised pretraining of the SimCLR model with a Resnet18 base
### Supporting Modules
- `ndpi_slide.py` - contains class wrapper for OpenSlide object for manipulating .ndpi files based on https://github.ic.ac.uk/jms3/sirius_red
- `data_augment.py` - contains class for applying affine and color augmentation during training.
- `model_zoo.py` -  contains models based on Ilse et al's Multiple Instance Learning frameworks (with Attention and Max Pooling) [3], encoders including SimCLR [4], SE-Resnet [5]
- `dataset_generic.py` -  contains classes for two dataloading frameworks, the one for generating a single bag per WSI and the second for training and testing with multiple inference.
- `Reinhard.py` -  code for application of Reinhard stain transfer [6]
- `utils.py` and `training.py` contain generic utility and model training and evaluation functions. 
- `ssc_utils.py` -  contains models and training functions for application of the Stain Standardisation Capsule based on https://github.com/Zhengyushan/ssc. [7]

### Analysis
- `test_set_summary.ipynb` - notebook that loads pretrained models and generates attention visualisations for each prediction

## References

[1] Berg S, Kutra D, Kroeger T, Straehle CN, Kausler BX, Haubold C, Schiegg M, Ales J, Beier T, Rudy M, Eren K. Ilastik: interactive machine learning for (bio) image analysis. Nature Methods. 2019 Dec;16(12):1226-32.

[2] L. Biewald, “Experiment Tracking with Weights and Biases,” Weights & Biases. [Online]. Available: http://wandb.com/. [Accessed: 29/08/2021].
Software available from wandb.com

[3] Ilse M, Tomczak J, Welling M. Attention-based deep multiple instance learning. InInternational conference on machine learning 2018 Jul 3 (pp. 2127-2136). PMLR.

[4] Chen T, Kornblith S, Norouzi M, Hinton G. A simple framework for contrastive learning of visual representations. InInternational conference on machine learning 2020 Nov 21 (pp. 1597-1607). PMLR.

[5] Hu J, Shen L, Sun G. Squeeze-and-excitation networks. InProceedings of the IEEE conference on computer vision and pattern recognition 2018 (pp. 7132-7141).

[6] Reinhard E, Adhikhmin M, Gooch B, Shirley P. Color transfer between images. IEEE Computer graphics and applications. 2001 Jul;21(5):34-41.

[7] Yushan Zheng, Zhiguo Jiang*, Haopeng Zhang, Fengying Xie, Dingyi Hu, Shujiao Sun, Jun Shi, and Chenghai Xue, Stain standardization capsule (SSC) for application-driven histopathological image normalization, IEEE Journal of Biomedical and Health Informatics, 2021, 25(2):337-347.
