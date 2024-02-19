# Segment Anything Model Can Not Segment Anything: Assessing AI Foundation Model’s Generalizability in Permafrost Mapping

## Overview
This repository contains the models and codes used in the work, "Segment Anything Model Can Not Segment Anything: Assessing AI Foundation Model’s Generalizability in Permafrost Mapping". This project delves into the workings of foundation models within the specialized field of terrain mapping. Specifically, we focus on the study of three terrain datasets: ice wedge polygon (IWP), retrogressive thaw Slump (RTS) and [EuroCrops](https://www.eurocrops.tum.de/index.html). The modeling and code within this repository have been implemented using [PyTorch](https://pytorch.org) and the [Detectron2](https://github.com/facebookresearch/detectron2) framework.

## Installation
The installation instructions for each package can be found in the following links:
1. [PyTorch](https://pytorch.org/get-started/locally/)
2. [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
3. [Segment Anyting](https://github.com/facebookresearch/segment-anything)
4. [MViTv2](https://github.com/facebookresearch/detectron2/tree/main/projects/MViTv2)

## Datasets
This project utilizes three datasets. For IWP and RTS datasets, both are provided by Maxar Technologies and are subject to sharing restrictions. The direct access to or distribution of this specific dataset is not permitted as per the agreement with Maxar Technologies. Researchers interested in using this dataset must obtain it directly from Maxar Technologies under their terms of use.

## Experiments
There are six experiments in this project. The experiments are listed below:

1. mvitv2: The model is trained using the [MViTv2](https://arxiv.org/abs/2112.01526) as a baseline. The <dataset_name> can be `iwp`, `rts` or `agr`. To train the model, execute the following script:

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

4. sam_maskrcnn_no_finetune: This experiment tests the performance of SAM with MViTv2. It acts as a real-world scenario where the user has a pre-trained model to provide the input prompt to SAM. The SAM model is untrained to test the performance as a foundation model. Use MViTv2 trained weights from experiment 1. To run the experiment, please use the following command:

```python
python experiments/sam_maskrcnn_no_finetune/sam_mvitv2_no_finetune.py --dataset <dataset_name> --mvitv2 <path_to_mvitv2_model_weights> 
```

5. sam_maskrcnn_with_finetune: This experiment tests the performance of a finetuned SAM model with MViTv2. It acts as a real-world scenario where the user has a pre-trained model to provide the input prompt to SAM. The SAM model is finetuned in this experiment. Use MViTv2 trained weights from experiment 1 and SAM trained weights from experiment 3. To run the experiment, please use the following command:

```python
python experiments/sam_maskrcnn_with_finetune/sam_mvitv2_with_finetune.py --dataset <dataset_name> --mvitv2 <path_to_mvitv2_model_weights> --sam <path_to_sam_model_weights>
```

6. sam_clip: this experiment reports the zero-shot learning result of SAM + CLIP for both datasets. For more information, please refer to the REAME.md in the `SAM+CLIP` folder.


## Predictions
To generate the predictions for the six experiments in the paper, please use the following command:

```python
python utils/generate_predictions_for_paper.py --dataset <dataset_name> --file <file_name>
```

