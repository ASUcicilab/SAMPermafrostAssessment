# %%
import os
from functools import lru_cache
from random import randint
from typing import Any, Callable, Dict, List, Tuple

import clip
import cv2
import numpy as np
import PIL
import torch
import pycocotools.mask
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, default='data/rts/rts_val.json', help='Input label file.')
parser.add_argument('-c', '--clip', type=bool, default=False, help='Whether to use CLIP model.')
parser.add_argument('--prompt', type=str, default='thaw slump', help='Prompt for CLIP model.')
parser.add_argument('--n_candidates', type=int, default=100, help='Maximum number of candidates to be selected from SAM.')
parser.add_argument('--n_points', type=int, default=32, help='The number of points to be sampled along one side of the image. The total number of points is points_per_side**2.')
parser.add_argument('--gpu', type=int, default=5, help='the index of GPU to use (0-5).')
args = parser.parse_args()

iDev = args.gpu
torch.set_num_threads(24)

CHECKPOINT_PATH = os.path.join(os.path.expanduser("~"), ".cache", "SAM")
CHECKPOINT_NAME = "sam_vit_h_4b8939.pth"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
MODEL_TYPE = "default"
LABEL_FILE = args.input
MAX_WIDTH = MAX_HEIGHT = 1024
TOP_K_OBJ = args.n_candidates
POINT_NUM = args.n_points
SCORE_NAME = "clip_score" if args.clip else "stability_score"
# SCORE_NAME = "clip_score"
# SCORE_NAME = "stability_score"
# SCORE_NAME = "static_score"
PROMPT = args.prompt
RES_FILE_NAME = 'sam.%s.%s.c%d.p%d.json' % (PROMPT.replace(' ', '_'), SCORE_NAME, TOP_K_OBJ, POINT_NUM)
device = torch.device("cuda", index=iDev) if torch.cuda.is_available() else torch.device("cpu")

# %%
@lru_cache
def load_mask_generator() -> SamAutomaticMaskGenerator:
    import urllib
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)
    checkpoint = os.path.join(CHECKPOINT_PATH, CHECKPOINT_NAME)
    if not os.path.exists(checkpoint):
        urllib.request.urlretrieve(CHECKPOINT_URL, checkpoint)
    sam = sam_model_registry[MODEL_TYPE](checkpoint=checkpoint).to(device)
    mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=POINT_NUM)
    return mask_generator


@lru_cache
def load_clip(
    name: str = "ViT-B/32",
) -> Tuple[torch.nn.Module, Callable[[PIL.Image.Image], torch.Tensor]]:
    model, preprocess = clip.load(name, device=device)
    return model.to(device), preprocess


def adjust_image_size(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    if height > width:
        if height > MAX_HEIGHT:
            height, width = MAX_HEIGHT, int(MAX_HEIGHT / height * width)
    else:
        if width > MAX_WIDTH:
            height, width = int(MAX_WIDTH / width * height), MAX_WIDTH
    image = cv2.resize(image, (width, height))
    return image


@torch.no_grad()
def get_score(crop: PIL.Image.Image, texts: List[str]) -> torch.Tensor:
    model, preprocess = load_clip()
    preprocessed = preprocess(crop).unsqueeze(0).to(device)
    tokens = clip.tokenize(texts).to(device)
    logits_per_image, _ = model(preprocessed, tokens)
    similarity = logits_per_image.softmax(-1).cpu()
    return similarity[0, 0]


def crop_image(image: np.ndarray, mask: Dict[str, Any]) -> PIL.Image.Image:
    x, y, w, h = mask["bbox"]
    masked = image * np.expand_dims(mask["segmentation"], -1)
    crop = masked[y : y + h, x : x + w]
    if h > w:
        top, bottom, left, right = 0, 0, (h - w) // 2, (h - w) // 2
    else:
        top, bottom, left, right = (w - h) // 2, (w - h) // 2, 0, 0
    # padding
    crop = cv2.copyMakeBorder(
        crop,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    crop = PIL.Image.fromarray(crop)
    return crop


def get_texts(query: str) -> List[str]:
    return [f"a picture of {query}", "a picture of background"]

def draw_masks(
    image: np.ndarray, masks: List[np.ndarray], alpha: float = 0.7
) -> np.ndarray:
    for mask in masks:
        color = [randint(127, 255) for _ in range(3)]

        # draw mask overlay
        colored_mask = np.expand_dims(mask["segmentation"], 0).repeat(3, axis=0)
        colored_mask = np.moveaxis(colored_mask, 0, -1)
        masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
        image_overlay = masked.filled()
        image = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

        # draw contour
        contours, _ = cv2.findContours(
            np.uint8(mask["segmentation"]), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
    return image

def filter_masks(
    image: np.ndarray,
    masks: List[Dict[str, Any]],
    predicted_iou_threshold: float = .9,
    stability_score_threshold: float = .8,
    query: str = '',
    clip_threshold: float = 0,
) -> List[Dict[str, Any]]:
    filtered_masks: List[Dict[str, Any]] = []

    for mask in sorted(masks, key=lambda mask: mask["area"])[-TOP_K_OBJ:]:
        try:
            clip_score = get_score(crop_image(image, mask), get_texts(query))
        except Exception as e:
            continue
        
        if (
            mask["predicted_iou"] < predicted_iou_threshold
            or mask["stability_score"] < stability_score_threshold
            or image.shape[:2] != mask["segmentation"].shape[:2]
            or query and clip_score < clip_threshold
        ):
            continue

        mask['clip_score'] = clip_score.item()
        filtered_masks.append(mask)

    return filtered_masks

# %%
# dataset extended from torch dataset
from torch.utils.data import Dataset
import json

class IWP(Dataset):
    def __init__(self, input_file):
        with open(input_file) as f:
            self.dataset = json.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

# %%
def prediction(**args):
    """
    generate prediction for SAM and transfer the prediction to COCO's format
    """
    dataset = IWP(args['input_file'])
    tmp_file = '.tmp.png'

    sam_pred_in_coco_format = []

    # create output folder
    output_dir = '{}/{}'.format(args['output_dir'], args['dataset'])
    sam_dir = '{}/sam'.format(output_dir)
    pred_dir = '{}/pred'.format(output_dir)
    if not os.path.exists(sam_dir):
        os.makedirs(sam_dir)
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    for idx, data in enumerate(dataset):
        # show progress
        print('Processing image {}/{}'.format(idx, len(dataset)))

        img_path = f'data/rts/{data["img"]}'
        img_id = data['img'].split('.')[0].split('/')[-1]

        # load image
        mask_generator = load_mask_generator()
        image = np.load(img_path)
        image = PIL.Image.fromarray((image[..., :3] * 255).astype(np.uint8)).convert('RGB')
        image.save(tmp_file)
        image = cv2.imread(tmp_file, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = adjust_image_size(image)
        masks = mask_generator.generate(image)
        
        masks = filter_masks(
            image, masks, 
            predicted_iou_threshold=0.9,
            stability_score_threshold=0.95,
            clip_threshold=0,
            query=PROMPT
        )

        for mask in masks:
            pred_sam = {}
            pred_sam['image_id'] = img_id
            pred_sam['category_id'] = 0
            pred_sam['bbox'] = mask['bbox']
            segm = mask['segmentation']

            # convert the mask to COCO's compressed RLE format
            segm = segm.astype(np.uint8)
            segm = pycocotools.mask.encode(np.asarray(segm, order="F"))
            segm['counts'] = segm['counts'].decode('ascii')

            pred_sam['segmentation'] = segm
            pred_sam['score'] = mask[SCORE_NAME]

            sam_pred_in_coco_format.append(pred_sam)


        sam_image = draw_masks(image, masks)
        sam_image = PIL.Image.fromarray(sam_image)
        sam_image.save('{}/{}.png'.format(sam_dir, img_path.split('/')[-1].split('.')[0]))

        masks = filter(lambda mask: mask['clip_score'] > 0.8, masks)
        pred_image = draw_masks(image, masks)
        pred_image = PIL.Image.fromarray(pred_image)
        pred_image.save('{}/{}.png'.format(pred_dir, img_path.split('/')[-1].split('.')[0]))

    # save the result as coco result format
    with open('{}/{}'.format(output_dir, RES_FILE_NAME), 'w') as f:
        json.dump(sam_pred_in_coco_format, f)


# %%
def summarize(eval_obj):
    '''
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    '''
    def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
        p = eval_obj.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap==1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = eval_obj.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,:,aind,mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = eval_obj.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,aind,mind]
        if len(s[s>-1])==0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s>-1])
        print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
        return mean_s
    
    _summarize(1, iouThr=.5, maxDets=eval_obj.params.maxDets[2])
    _summarize(0, iouThr=.5, maxDets=eval_obj.params.maxDets[2])

def generate_evaluation_metric(**args):
    """
    this function is used to generate the evaluation metric for SAM
    it transfers the ground truth to COCO's format and call COCO's API to evaluate the segmentation quality
    """
    # load the test dataset ground truth and transfer it to COCO's format
    dataset = json.load(open('data/{}/{}_val.json'.format(args['dataset'], args['dataset'])))
    coco_gt = {}
    coco_gt['images'] = []
    coco_gt['annotations'] = []
    ann_id = 0
    
    # iterate through the dataset and transfer it to COCO's format
    for data in dataset:

        img_path = f'data/rts/{data["img"]}'
        img_id = data['img'].split('.')[0].split('/')[-1]
        msks = np.load(img_path.replace('npy', 'mask.npy'))

        image = {}
        image['id'] = img_id
        image['file_name'] = img_path
        image['width'] = data['width']
        image['height'] = data['height']
        coco_gt['images'].append(image)

        # annotations
        for i, (box, ctg) in enumerate(zip(data['box'], data['label'])):
            ann_coco = {}
            ann_coco['id'] = ann_id
            ann_id += 1
            
            segm = msks[i, ...]
            segm = (segm > 0).astype(np.uint8)
            segm = pycocotools.mask.encode(np.asarray(segm, order="F"))
            segm['counts'] = segm['counts'].decode('ascii')

            ann_coco['image_id'] = img_id
            ann_coco['category_id'] = ctg
            ann_coco['segmentation'] = segm
            ann_coco['bbox'] = box
            ann_coco['area'] = (box[2] - box[0]) * (box[3] - box[1])
            ann_coco['iscrowd'] = 0
            coco_gt['annotations'].append(ann_coco)

        # categories
        coco_gt['categories'] = []
        for id in range(args['num_classes']):
            category = {}
            category['id'] = id
            category['name'] = args['dataset'] + '_' + str(id)
            category['supercategory'] = args['dataset'] + '_' + str(id)
            coco_gt['categories'].append(category)

    # save the coco_gt to a json file
    with open('results/{}/coco_gt.json'.format(args['dataset']), 'w') as f:
        json.dump(coco_gt, f)

    # call coco api to evaluate the segmentation quality
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco_gt = COCO('results/{}/coco_gt.json'.format(args['dataset']))
    coco_dt = coco_gt.loadRes('results/{}/{}'.format(args['dataset'], RES_FILE_NAME))
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    # coco_eval.params.maxDets[2] = TOP_K_OBJ
    # coco_eval.evaluate()
    # coco_eval.accumulate()
    # coco_eval.summarize()
    # print(coco_eval.stats) # 0.615

    coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
    coco_eval.params.maxDets[2] = TOP_K_OBJ
    coco_eval.evaluate()
    coco_eval.accumulate()
    summarize(coco_eval)
    # print(coco_eval.stats) # 0.371

# %%
prediction(input_file=LABEL_FILE, output_dir='results', dataset='rts')
generate_evaluation_metric(dataset='rts', num_classes=1)
