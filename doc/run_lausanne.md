

# Run Lausanne dataset
## Installation
### A. Clone repository

```bash
# clone repository
git clone https://github.com/MrZoiya/CityGaussian.git
cd CityGaussian
```

### B. Create virtual environment

```bash
# create virtual environment
conda create -yn gspl python=3.9 pip
conda activate gspl
```

### C. Install PyTorch
* Tested on `PyTorch==2.6.1`
* You must install the one match to the version of your nvcc (nvcc --version)
* For CUDA 12.4

  ```bash
  pip install -r requirements/pyt251_cu124.txt
  ```

### D. Install requirements

```bash
pip install -r requirements.txt
```

### E. Install additional package for CityGaussian

```bash
pip install -r requirements/CityGS.txt
```
Note that here we use modified version of Trim2DGS rasterizer, so as to resolve [impulse noise problem](https://github.com/hbb1/2d-gaussian-splatting/issues/174) under street views. This version also avoids interference from out-of-view surfels.

## Prepare COLMAP dataset

### A. Create image.txt and cameras.txt

In a data folder, add the image folder the cameras.xml file. The use the following command to generate the `image.txt` and `cameras.txt` files:

```bash
python tools/lausanne/convert_lausanne.py --xml_path path/to/cameras.xml --image_path path/to/images --output_path path/to/output
```

### B. Prepare COLMAP dataset

Then run the notebook in `notebooks/context2colmap` to generate the COLMAP dataset. Colmap was built from the most
recent version so version 3.12.0.
This notebook will print the COLMAP command to run. Take care to change the different *_path variables to your own paths.

The command should look like this:

```bash
  colmap feature_extractor \
    --database_path=path/to/colmap.db \
    --image_path=path/to/images \
    --ImageReader.camera_model=OPENCV
    
  Adjust the database.db to input the right intrinsics and extrinsics in the notebook
   
  colmap vocab_tree_matcher \
    --database_path=path/to/colmap.db \
    --VocabTreeMatching.vocab_tree_path=path/to/vocab_tree.bin
    
  colmap point_triangulator \
    --database_path=path/to/colmap.db \
    --image_path=path/to/images \
    --input_path=path/to/sparse \
    --output_path=path/to/sparse
    
  colmap bundle_adjuster \
    --input_path path/to/sparse \
    --output_path path/to/bundle_adjusted
```
This bundle adjusted folder will be used as the sparse folder in the next steps. We renamed it to sparse so that it is compatible with the rest of the code.

## Run Lausanne dataset

The commands to run Lausanne dataset are available in `notebooks/generate_commands.ipynb`. Before running the commands, 
make sure to change the paths to your own paths. You also need to get Depth Anything V2 to run the depth regularization.
```bash
# clone the repo.
git clone https://github.com/DepthAnything/Depth-Anything-V2 utils/Depth-Anything-V2

# NOTE: do not run `pip install -r utils/Depth-Anything-V2/requirements.txt`

# download the pretrained model `Depth-Anything-V2-Large`
mkdir utils/Depth-Anything-V2/checkpoints
wget -O utils/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true"
```

You must also create your own `config.yaml` files in the `configs` folder. Create one for the coarse approximation and one for the fine approximation.
You can use the file the `configs/lausanne_train.yaml` and `configs/lausanne_9000_coarse.yaml` as a template.

## Visualization

## A. Render video

Use the following command to render a video of Lausanne dataset:

```bash
python tools/render_traj.py --output_path outputs/$NAME --filter --train 
```

## B. Visualize in viewer
You can visualize the Lausanne dataset in the viewer by running the following command:

```bash
python viewer.py path/to/your/output
```

