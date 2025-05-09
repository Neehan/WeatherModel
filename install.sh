# install conda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

# initialize conda
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh

# Ensure conda is initialized in the script
source ~/miniconda3/etc/profile.d/conda.sh

# Create a new conda environment named "weather" with Python 3.10
conda create -n weather python=3.10 -y

# Activate the new environment
conda activate weather

# Install PyTorch, transformers and datasets in the new environment
conda install pytorch transformers datasets torch-lr-finder python-dotenv matplotlib pandas -c pytorch -y
