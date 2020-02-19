# Visual-Prior-VGPL

PyTorch implementation for the visual prior componenet (i.e. percetion module) of the Visually Grounded Physics Learner (VGPL).
Given visual obseravtions, the visual prior proposes their corresponding particle representations, in the form of particle positions and groupings.

## Prerequisites

- Python 3
- PyTorch 1.0 or higher, with NVIDIA CUDA Support
- Other required packages in `requirements.txt`

## Code overview

### Helper files

`config.py` contains all configurations used for model training, model evaluation and output generation.

`dataset.py` contains helper functions for loading and standardizing data and related variables. Note that paths to data directories is specified in the `_DATA_DIR` variable in this file, not in `config.py`.

`loss.py` contains helper functions for calculating Chamfer loss in different settings (e.g. in a single frame, across a time sequence, etc.).

`model.py` implements the neural network model used for prediction.

### Main files

The following files can be run directly; see "Training and evaluation" section for more details.

`train.py` trains a model that could convert input observations into their particle representations.

`eval.py` evaluates a trained model by visualizing its predictions, and/or stores the output predictions in `.h5` format.

## Training and evaluation

See `config.py` for more details on customizable configurations.

To train the model:

`python train.py --set loss_type chamfer dataset RigidFall`

To debug (by overfitting model on small batch of data):

`python train.py --set loss_type chamfer dataset RigidFall debug True`

To evaluate a trained model and generate outputs:

`python eval.py --set loss_type chamfer dataset RigidFall n_frames 4 n_frames_eval 10 load_path [path]`