### This codebase is built on top of https://github.com/Ghostish/Open3DSOT, with references from https://github.com/HaozheQi/P2B and https://github.com/haooozi/OSP2B. Our Codebase was setup on GCP with 100gb storage for this project and NVIDIA L4 GPU(24GB RAM) and g2-standard-8 8vcpus 32GB RAM

## Parts of the code fully implemented by us
models/customtransformer.py -> Akshay (referred ideas from OSP2B transformer)
models/backbone/pointnet2.py -> Roshini (referred ideas from P2B backbone)
visualization/3dbox_to_cloud.py -> Akshay (referred ideas https://github.com/zzzxxxttt/simple_kitti_visualization/tree/master)
visualization/im_to_vid.py -> Roshini 
visualization/vid2gif.py -> Roshini
visualization/vis_single_pc.py -> Roshini (modified from 3dbox_to_cloud.py)
test_all.sh -> Akshay
transformers-exp.ipynb (experiment notebook for transformerss code, has similar code to customtransformer.py)-> Akshay

## below are some lines of codes that we changed/wrote in different files
main.py -> lines 41-43, 74-88, 102-108, 116-130 -> Akshay
Dataset.py -> lines 306-320, line 269 -> Akshay
models/p2b.py -> lines 8-9, 21-22, 66-68, 72, 81-82, 168-180 ->Roshini
models/base_model.py -> 15-16, 69, 87-98 -> Roshini
## Apart from the above snippets, there are a lot of modifications made (which were untraced during the process) inorder to integrate the Open3DSOT code snippits with P2B Dataloader and our snippets
## Apart from the above repositories and codes, we also tried setting up couple of other repositories initially and didn't get good results, so not providing any of those changes here. We also spent decent amount of time researching on Google cloud and setting up everything and gathering required permissions


### installation

# environment setup
cd 674-proj
conda create -n myenv  python=3.8
conda activate myenv

# install pytorch with cuda support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# install requirements
pip install -r requirement.txt

## Download KITTI Velodyne dataset, calib and label_02 files and unzip them and place them as follows
[Parent Folder]
--> [calib]
    --> {0000-0020}.txt
--> [label_02]
    --> {0000-0020}.txt
--> [velodyne]
    --> [0000-0020] folders with velodynes .bin files

## Training the model on Car category
CUDA_VISIBLE_DEVICES=0 python main.py  --cfg cfgs/P2B_Car.yaml  --batch_size 64 --epoch 60 --category_name Car

## Testing model on Car category
python main.py  --cfg cfgs/P2B_Car.yaml  --checkpoint ./ckpts/car-31.ckpt  --test --category_name Car

## To run tensorboard after training
pip install tensorboard
tensorboard --logdir=./ --port=6006

## trained model checkpoints present in 674-proj/ckpts/