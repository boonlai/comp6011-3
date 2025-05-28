# COMP6011 - Task 3 DINOv2 Fine-tuning
Paramat Boonlai (18196201)

### Requirements:
- [Conda](https://www.anaconda.com/docs/getting-started/miniconda/install)
- [CUDA Toolkit 12.6](https://developer.nvidia.com/cuda-downloads)
- [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) extracted to `ptb-xl/` (for training only)


## Getting Started

This repository should already come with a pre-trained checkpoint file `finetuned_model_7.pt`, so dataset generation and training aren't necessary.

### Environment Setup

1. Run `conda env create -f environment.yml`
2. Switch to the environment `conda activate comp6011-task3`
3. Additionally install `pip install imbalanced-learn wfdb roboflow`

### Generating Dataset

1. Make sure the PTB-XL dataset is located at `ptb-xl/`, so `ptb-xl/ptbxl_database.csv` exists
2. Run all cells in the notebook
3. Upload the generated images to [Roboflow](https://roboflow.com/), or split it yourself to:
    - `data/train/[class]/{image}`
    - `data/valid/[class]/{image}`
    - `data/test/[class]/{image}`
    - where the classes are: `1AVB`, `AFIB`, `AFLT`, `LBBB`, `RBBB`, `NORM`, `OTHERS`

### Training

1. If you are using [Roboflow](https://roboflow.com/), modify `rf_workspace` `rf_project` `rf_version`
2. Delete the existing `finetuned_model_7.pt` or it will simply evaluate it against the test data
3. Run the training script `python train.py`

### Predicting

Run `python predict.py [target]`.

The `[target]` can either be a file or a directory. For example:
- `python predict.py validation`
- `python predict.py test`
- `python predict.py test/test01/test01.npy`

If a directory is given, `.dat` and `.npy` files will be searched recursively.

