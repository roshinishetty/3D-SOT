# Self-Driving-Cars - Object-Tracking
Please check [Slides](<./Slides.pdf> "Slides") and [Project Report](<./Report.pdf> "Project Report") for more details and figures

## Sample Visualiztion 
Point Cloud Sequence with 3D BBox tracked and its Corresponding Video Sequence

<img src="https://github.com/Akhy999/Self-Driving-Cars---Object-Tracking/blob/main/figs/output2.gif" width="760" />

### This codebase is built on top of https://github.com/Ghostish/Open3DSOT, with references from https://github.com/HaozheQi/P2B and https://github.com/haooozi/OSP2B. Our Codebase was setup on GCP with 100gb storage for this project and NVIDIA L4 GPU(24GB RAM) and g2-standard-8 8vcpus 32GB RAM

# Installation

## environment setup
cd 3dsot
conda create -n myenv  python=3.8
conda activate myenv

## install pytorch with cuda support
```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117```

## install requirements
```pip install -r requirement.txt```

## Download KITTI Velodyne dataset, calib and label_02 files and unzip them and place them as follows
```[Parent Folder]
--> [calib]
    --> {0000-0020}.txt
--> [label_02]
    --> {0000-0020}.txt
--> [velodyne]
    --> [0000-0020] folders with velodynes .bin files
```

## Training the model on Car category
```CUDA_VISIBLE_DEVICES=0 python main.py  --cfg cfgs/P2B_Car.yaml  --batch_size 64 --epoch 60 --category_name Car```

## Testing model on Car category
```python main.py  --cfg cfgs/P2B_Car.yaml  --checkpoint ./ckpts/car-31.ckpt  --test --category_name Car```

## To run tensorboard after training
```
pip install tensorboard
tensorboard --logdir=./ --port=6006
```

## trained model checkpoints present in 3dsot/ckpts/
