## Introduction

BlumNet2 is the sucessor of [BlumNet](https://github.com/cong-yang/BlumNet). 
* Improved with graph loss.  

## Citation
TBD


## Usage
1. Prepare python environments by  
    ```bash
    pip install -r requirements.txt
    ```

2. Build [deformable transformer](https://github.com/fundamentalvision/Deformable-DETR)  
    ```bash
    cd ./models/ops
    sh ./make.sh
    ```

3. Download pretrained backbone 
    ```bash
    mkdir -p models/pretrained
    cd models/pretrained
    # download swin-transformer weights
    wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth
    wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth
    ```

4. Prepare dataset  
    Please download sk1491 from [GoogleDrive](https://drive.google.com/file/d/11ya3dDYnbiUEAElz9aZVnf6aN5uTg77F/view?usp=sharing) and organize them as following:
    ```
    code_root/
    └── data/
        └── sk1491/
            ├── test/
            ├── train/
                ├── im
                ├── gt
                └── train_pair.lst
    ```

5. Train model  
    ```bash
    export WANDB_API_KEY="xxx"
    export WANDB_MODE="online"
    python train.py

    ```


## Results

F1 Score  
| Backbone | sk506 | sk1491 | WH-SYMMAX | SymPASCAL |
| ----- | ----- | ----- | ----- | ----- |
| SwinBase |     | 0.833  |    |     |


