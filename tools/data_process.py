import argparse
import json
import os

import cv2
import numpy as np
from tqdm import tqdm

from refer import REFER

# 参数解析
parser = argparse.ArgumentParser(description='Data preparation')
parser.add_argument('--data_root', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--dataset',
                    type=str,
                    choices=['refcoco', 'refcoco+', 'refcocog', 'refclef'],
                    default='refcoco')
parser.add_argument('--split', type=str, default='umd')
parser.add_argument('--generate_mask', action='store_true')
args = parser.parse_args()

img_path = os.path.join(args.data_root, 'images', 'train2014')
h, w = (416, 416)

# 加载 REFER 数据
refer = REFER(args.data_root, args.dataset, args.split)

print('dataset [%s_%s] contains: ' % (args.dataset, args.split))
ref_ids = refer.getRefIds()
image_ids = refer.getImgIds()
print('%s expressions for %s refs in %s images.' % (len(refer.Sents), len(ref_ids), len(image_ids)))

print('\nAmong them:')
if args.dataset == 'refclef':
    if args.split == 'unc':
        splits = ['train', 'val', 'testA', 'testB', 'testC']
    else:
        splits = ['train', 'val', 'test']
elif args.dataset in ['refcoco', 'refcoco+']:
    splits = ['train', 'val', 'testA', 'testB']
elif args.dataset == 'refcocog':
    splits = ['train', 'val', 'test']

for split in splits:
    ref_ids = refer.getRefIds(split=split)
    print('%s refs are in split [%s].' % (len(ref_ids), split))


# 类别索引调整函数
def cat_process(cat):
    if 1 <= cat <= 11:
        cat -= 1
    elif 13 <= cat <= 25:
        cat -= 2
    elif 27 <= cat <= 28:
        cat -= 3
    elif 31 <= cat <= 44:
        cat -= 5
    elif 46 <= cat <= 65:
        cat -= 6
    elif cat == 67:
        cat -= 7
    elif cat == 70:
        cat -= 9
    elif 72 <= cat <= 82:
        cat -= 10
    elif 84 <= cat <= 90:
        cat -= 11
    return cat


# bbox 处理为 [x_min, y_min, x_max, y_max]
def bbox_process(bbox):
    x_min = int(bbox[0])
    y_min = int(bbox[1])
    x_max = x_min + int(bbox[2])
    y_max = y_min + int(bbox[3])
    return [x_min, y_min, x_max, y_max]


# 数据准备函数
def prepare_dataset(dataset, splits, output_dir, generate_mask=False):
    ann_path = os.path.join(output_dir, 'anns', dataset)
    mask_path = os.path.join(output_dir, 'masks', dataset)
    os.makedirs(ann_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)

    for split in splits:
        dataset_array = []
        ref_ids = refer.getRefIds(split=split)
        print('Processing split:{} - Len: {}'.format(split, len(ref_ids)))
        for i in tqdm(ref_ids):
            ref_dict = {}
            refs = refer.Refs[i]
            bboxs = refer.getRefBox(i)
            sentences = refs['sentences']
            image_urls = refer.loadImgs(image_ids=refs['image_id'])[0]
            cat = cat_process(refs['category_id'])
            image_file_name = image_urls['file_name']

            # 特殊图像跳过
            if dataset == 'refclef' and image_file_name in ['19579.jpg', '17975.jpg', '19575.jpg']:
                continue

            box_info = bbox_process(bboxs)

            ref_dict['bbox'] = box_info
            ref_dict['cat'] = cat
            ref_dict['segment_id'] = i
            ref_dict['img_name'] = image_file_name

            # 生成 mask
            if generate_mask:
                mask = refer.getMask(refs)['mask'] * 255
                cv2.imwrite(os.path.join(mask_path, str(i) + '.png'), mask)

            sent_dict = []
            for j, sent in enumerate(sentences):
                sent_dict.append({
                    'idx': j,
                    'sent_id': sent['sent_id'],
                    'sent': sent['sent'].strip()
                })

            ref_dict['sentences'] = sent_dict
            ref_dict['sentences_num'] = len(sent_dict)

            dataset_array.append(ref_dict)

        print('Dumping json file...')
        with open(os.path.join(ann_path, split + '.json'), 'w') as f:
            json.dump(dataset_array, f)


# 开始执行
prepare_dataset(args.dataset, splits, args.output_dir, args.generate_mask)
