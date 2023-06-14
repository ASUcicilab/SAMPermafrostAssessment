# GeoAI Foundation Models for Vision: A View from AI Augmented Terrain Mapping

## Overview
This repository contains the models and codes used in the paper, "GeoAI Foundation Models for Vision: A View from AI Augmented Terrain Mapping". This project delves into the workings of foundation models within the specialized field of terrain mapping. Specifically, we focus on the study of two terrain features: Ice Wedge Polygons (IWPs) and Retrogressive Thaw Slumps (RTS). The modeling and code within this repository have been implemented using [PyTorch](https://pytorch.org) and the [Detectron2](https://github.com/facebookresearch/detectron2) framework.

## Installation
The installation instructions for each package can be found in the following links:
1. [PyTorch](https://pytorch.org/get-started/locally/)
2. [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
3. [Segment Anyting](https://github.com/facebookresearch/segment-anything)
4. [MViTv2](https://github.com/facebookresearch/detectron2/tree/main/projects/MViTv2)

## Datasets
This project utilizes the following datasets, which can be accessed via the provided links:

1. [IWP Dataset]()
2. [RTS Dataset]()

In order to prepare these datasets for usage with the Detectron2 framework, a preprocessing step is necessary. This can be accomplished by executing the script as follows:

```Python
python data/preprocess.py --dataset <dataset_name>
```

In the above command, replace `<dataset_name>` with either `iwp` or `rts`, depending on the dataset you wish to preprocess.

Additionally, to generate data samples complete with bounding boxes and masks, execute the following script:

```Python
python data/generate_samples.py --dataset <dataset_name> --num_samples <num_samples>
```

## Experiments
There are six experiments in this project. The experiments are listed below:
1. mvitv2: The model is trained using the [MViTv2](https://arxiv.org/abs/2112.01526) as a baseline. To train the model, execute the following script:

```Python
python experiments/mvitv2/mvitv2.py --dataset <dataset_name>
```
2. sam_no_finetune: This experiment tests the performance of SAM before finetuing. It tests the performance of SAM as a foundation model. The prompt can be either `bbox` or `point`. To test the perfomrance of SAM, execute the following script:

```Python
python experiments/sam_no_finetune/sam_no_finetune.py --dataset <dataset_name> --prompt <prompt>
```

3. sam_with_finetune: This experiment finetunes the SAM model. The prompt can be either `bbox` or `point`. To finetune the SAM model, execute the following script:

```Python
python experiments/sam_with_finetune/sam_with_finetune.py --dataset <dataset_name> --prompt <prompt>
```

4. sam_maskrcnn_no_finetune: This experiment tests the performance of SAM with MViTv2. It acts as a real-world scenario where the user has a pre-trained model to provide the input prompt to SAM. The SAM model is untrained to test the performance as a foundation model. To run the experiment, please use the following command:

```python
python experiments/sam_maskrcnn_no_finetune/sam_mvitv2_no_finetune.py --dataset <dataset_name>
```

5. sam_maskrcnn_with_finetune: This experiment tests the performance of a finetuned SAM model with MViTv2. It acts as a real-world scenario where the user has a pre-trained model to provide the input prompt to SAM. The SAM model is finetuned in this experiment. To run the experiment, please use the following command:

```python
python experiments/sam_maskrcnn_with_finetune/sam_mvitv2_with_finetune.py --dataset <dataset_name>
```

6. sam_clip: this experiment reports the zero-shot learning result of SAM + CLIP for both datasets. For more information, please refer to the REAME.md in the `SAM+CLIP` folder.


## Predictions
To generate the predictions for the six experiments in the paper, please use the following command:

```python
python utils/generate_predictions.py --dataset <dataset_name> --file <file_name>
```

