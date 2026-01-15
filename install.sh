conda env create -f environment.yml
conda init
conda activate imitation_learning
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install 'numpy==1.26.4'
# pip install gym[classic_control,other]==0.25.2
pip install matplotlib
