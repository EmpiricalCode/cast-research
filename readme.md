## CAST Using SAM3D + Graph-based SDF Physics Correction

# Setup guide

Create conda env:

```
conda create --name autoseg python=3.11
conda activate autoseg
```

Exports:

```
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/usr/local/cuda-11.5 
```

Clone Grounded-Segment-Anything repo:

```
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything
mv Grounded-Segment-Anything/GroundingDINO GroundingDINO
mv Grounded-Segment-Anything/segment_anything segment_anything
```

Install SAM:

```
python -m pip install -e segment_anything
```

Install PyTorch

```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

Install GroundingDINO

```
pip install --no-build-isolation -e GroundingDINO
```

Install diffusers:

```
pip install --upgrade diffusers[torch]
```

Install RAM:

```
git clone https://github.com/xinyu1205/recognize-anything.git
pip install -r ./recognize-anything/requirements.txt
pip install -e ./recognize-anything/
```

Correct opencv, numpy, and transformers:

```
pip install opencv-python==4.8.1.78
pip install numpy==1.26.4
pip install transformers==4.35.2  
```

Download model weights:

```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/ram_swin_large_14m.pth
wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/tag2text_swin_14m.pth
```