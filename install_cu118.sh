# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate lhm

# install torch 2.3.0
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
pip install -U xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu118

# install dependencies
pip install -r requirements.txt

# install from source code to avoid the conflict with torchvision
pip uninstall basicsr -y
pip install git+https://github.com/XPixelGroup/BasicSR

cd ..
# install pytorch3d
# pip install "git+https://github.com/facebookresearch/pytorch3d.git"
conda install -y https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py310_cu118_pyt231.tar.bz2

# install sam2
pip install git+https://github.com/hitsz-zuoqi/sam2/

# or
# git clone --recursive https://github.com/hitsz-zuoqi/sam2
# pip install ./sam2

# install diff-gaussian-rasterization
module load gcc cuda/11.8
pip install --no-build-isolation git+https://github.com/ashawkey/diff-gaussian-rasterization.git


# or
# git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
# pip install ./diff-gaussian-rasterization

# install simple-knn
pip install git+https://github.com/camenduru/simple-knn/

# or
# git clone https://github.com/camenduru/simple-knn.git
# pip install ./simple-knn