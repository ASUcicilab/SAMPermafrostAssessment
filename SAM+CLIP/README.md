# SAM + CLIP experiments

This repository contains the code that examines the how well foundation models (SAM+CLIP) can be applied on features like retrogressive thaw slumps (RTS) and ice wedge polygons (IWP).

[SAM (Segment Anything Model)](https://ai.facebook.com/blog/segment-anything-foundation-model-image-segmentation/) is a foundation model released by Meta aims to resolve downstream segmentation tasks with prompt engineering, such as foreground/background points, bounding box, and mask.

[CLIP](https://openai.com/research/clip) is utilized to measure the correspondence between a given image and a word or phrase.

## Setup environment
[Anaconda](https://www.anaconda.com/) is required before start setup.
```bash
conda create -n sam_clip python=3.8
conda activate sam_clip
pip install -r requirements.txt
```

## Run experiments
- Obtain the mAP results of IWP that relies on SAM only:
    ```bash
    python eval_iwp.py 
    ``` 
- Obtain the mAP results of IWP that relies on SAM + CLIP:
    ```bash
    python eval_iwp.py --clip
    ``` 

- Obtain the mAP results of RTS that relies on SAM only:
    ```bash
    python eval_rts.py 
    ``` 
- Obtain the mAP results of RTS that relies on SAM + CLIP:
    ```bash
    python eval_rts.py --clip
    ``` 

