## Install environment

You can configure the required environment for the code based on the following instructions:

1. First, configure a virtual environment with Python version 3.10:

    ```bash
    conda create -n UWNet -y python=3.10
    conda activate UWNet
    ```

2. Install the dependency libraries:

    ```bash
    pip install ultralytics
    pip install einops
    pip install timm
    pip install dill
    ```

3. Install selective_scan:

    ```bash
    cd ultralytics/nn/extra_modules/selective_scan/ && pip install .
    ```

## Train

We have already configured the UWNet training file, and you can run it directly.

    ```bash
    cd UWNet-main && python train.py
    ```

## Val

We also provide the model training script, and you can run it directly.

    ```bash
    cd UWNet-main && python val.py
    ```

## Acknowledgement

    We extend our deepest gratitude to the Ultralytics、 Mamba-YOLO、 and StarNet(Rewrite the Stars) teams for their outstanding contributions to the open-source community.
