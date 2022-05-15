# Install dependencies
pip install -r requirements_final.txt
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

# Ilastik installation
wget https://files.ilastik.org/ilastik-1.3.3post3-Linux.tar.bz2
tar xjf ilastik-1.*-Linux.tar.bz2

# Copy pretrained model to Ilastik repo
cp Ilastik_pixel_segmentation.ilp ilastik-1.*-Linux