import os
from typing import List, Union

import cv2
import lmdb
import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import Dataset

from .simple_tokenizer import SimpleTokenizer as _Tokenizer

info = {
    'refcoco': {
        'train': 42404,
        'val': 3811,
        'val-test': 3811,
        'testA': 1975,
        'testB': 1810
    },
    'refcoco+': {
        'train': 42278,
        'val': 3805,
        'val-test': 3805,
        'testA': 1975,
        'testB': 1798
    },
    'refcocog_u': {
        'train': 42226,
        'val': 2573,
        'val-test': 2573,
        'test': 5023
    },
    'refcocog_g': {
        'train': 44822,
        'val': 5000,
        'val-test': 5000
    },
    'refcoco_mixed': {
        'train': 126908, # 42404+42278+42226=126908
        'val': 10189, # 3811+3805+2573=10189
    }
}
_tokenizer = _Tokenizer()


def tokenize(texts: Union[str, List[str]],
             context_length: int = 77,
             truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token]
                  for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}"
                )
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


class RefDataset(Dataset):
    def __init__(self, lmdb_dir, mask_dir, dataset, split, mode, input_size,
                 word_length, task='ris'):
        super(RefDataset, self).__init__()
        self.lmdb_dir = lmdb_dir
        self.mask_dir = mask_dir
        self.dataset = dataset
        self.split = split
        self.mode = mode
        self.task = task  # 'ris' or 'rec'
        self.input_size = (input_size, input_size)
        self.word_length = word_length
        self.mean = torch.tensor([0.48145466, 0.4578275,
                                  0.40821073]).reshape(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258,
                                 0.27577711]).reshape(3, 1, 1)
        self.length = info[dataset][split]
        self.env = None

    def _init_db(self):
        self.env = lmdb.open(self.lmdb_dir,
                             subdir=os.path.isdir(self.lmdb_dir),
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_pyarrow(txn.get(b'__len__'))
            self.keys = loads_pyarrow(txn.get(b'__keys__'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Delay loading LMDB data until after initialization: https://github.com/chainer/chainermn/issues/129
        if self.env is None:
            self._init_db()
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        ref = loads_pyarrow(byteflow)
        # img
        ori_img = cv2.imdecode(np.frombuffer(ref['img'], np.uint8),
                               cv2.IMREAD_COLOR)
        img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        img_size = img.shape[:2]
        # mask
        seg_id = ref['seg_id']
        mask_dir = os.path.join(self.mask_dir, str(seg_id) + '.png')
        
        # bbox (for REC task) - 从ref中获取bbox，格式为 [x, y, w, h]
        # 如果ref中有bbox就用，否则从mask计算
        bbox = ref.get('bbox', None)
        
        # sentences
        idx = np.random.choice(ref['num_sents'])
        sents = ref['sents']
        # transform
        mat, mat_inv = self.getTransformMat(img_size, True)
        img = cv2.warpAffine(
            img,
            mat,
            self.input_size,
            flags=cv2.INTER_CUBIC,
            borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255])
        if self.mode == 'train':
            # mask transform
            mask = cv2.imdecode(np.frombuffer(ref['mask'], np.uint8),
                                cv2.IMREAD_GRAYSCALE)
            mask = cv2.warpAffine(mask,
                                  mat,
                                  self.input_size,
                                  flags=cv2.INTER_LINEAR,
                                  borderValue=0.)
            mask = mask / 255.
            # sentence -> vector
            sent = sents[idx]
            word_vec = tokenize(sent, self.word_length, True).squeeze(0)
            img, mask = self.convert(img, mask)
            
            # 处理bbox (用于REC任务)
            if self.task == 'rec':
                # 从mask中计算bbox (如果原始bbox不可用)
                if bbox is None:
                    # 从变换后的mask计算bbox
                    mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else mask
                    bbox_transformed = self._get_bbox_from_mask(mask_np)
                else:
                    # 变换原始bbox
                    bbox_transformed = self._transform_bbox(bbox, mat, img_size)
                # 归一化到0-1，格式为 [cx, cy, w, h]
                bbox_normalized = self._normalize_bbox(bbox_transformed, self.input_size)
                bbox_tensor = torch.tensor(bbox_normalized, dtype=torch.float32)
                return img, word_vec, mask, bbox_tensor
            
            return img, word_vec, mask
        elif self.mode == 'val':
            # sentence -> vector
            sent = sents[0]
            word_vec = tokenize(sent, self.word_length, True).squeeze(0)
            img = self.convert(img)[0]
            
            # 处理bbox (用于REC任务验证)
            if self.task == 'rec':
                # 读取mask来计算ground truth bbox
                mask = cv2.imread(mask_dir, flags=cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask = mask / 255.
                    gt_bbox = self._get_bbox_from_mask(mask)  # [x, y, w, h] 原始尺度
                else:
                    gt_bbox = [0, 0, 1, 1]
                params = {
                    'mask_dir': mask_dir,
                    'inverse': mat_inv,
                    'ori_size': np.array(img_size),
                    'gt_bbox': np.array(gt_bbox),  # 原始尺度的bbox
                }
                return img, word_vec, params
            
            params = {
                'mask_dir': mask_dir,
                'inverse': mat_inv,
                'ori_size': np.array(img_size)
            }
            return img, word_vec, params
        else:
            # sentence -> vector
            img = self.convert(img)[0]
            params = {
                'ori_img': ori_img,
                'seg_id': seg_id,
                'mask_dir': mask_dir,
                'inverse': mat_inv,
                'ori_size': np.array(img_size),
                'sents': sents
            }
            return img, params

    def getTransformMat(self, img_size, inverse=False):
        ori_h, ori_w = img_size
        inp_h, inp_w = self.input_size
        scale = min(inp_h / ori_h, inp_w / ori_w)
        new_h, new_w = ori_h * scale, ori_w * scale
        bias_x, bias_y = (inp_w - new_w) / 2., (inp_h - new_h) / 2.

        src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
        dst = np.array([[bias_x, bias_y], [new_w + bias_x, bias_y],
                        [bias_x, new_h + bias_y]], np.float32)

        mat = cv2.getAffineTransform(src, dst)
        if inverse:
            mat_inv = cv2.getAffineTransform(dst, src)
            return mat, mat_inv
        return mat, None

    def convert(self, img, mask=None):
        # Image ToTensor & Normalize
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        if not isinstance(img, torch.FloatTensor):
            img = img.float()
        img.div_(255.).sub_(self.mean).div_(self.std)
        # Mask ToTensor
        if mask is not None:
            mask = torch.from_numpy(mask)
            if not isinstance(mask, torch.FloatTensor):
                mask = mask.float()
        return img, mask

    def __repr__(self):
        return self.__class__.__name__ + "(" + \
            f"db_path={self.lmdb_dir}, " + \
            f"dataset={self.dataset}, " + \
            f"split={self.split}, " + \
            f"mode={self.mode}, " + \
            f"task={self.task}, " + \
            f"input_size={self.input_size}, " + \
            f"word_length={self.word_length}"
    
    def _get_bbox_from_mask(self, mask):
        """从mask中计算bbox，返回 [x, y, w, h] 格式"""
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        
        # 找到mask中非零区域
        if mask.max() <= 1:
            mask_binary = (mask > 0.5).astype(np.uint8)
        else:
            mask_binary = (mask > 127).astype(np.uint8)
        
        coords = np.where(mask_binary > 0)
        if len(coords[0]) == 0:
            # 如果mask为空，返回整图bbox
            h, w = mask.shape[:2]
            return [0, 0, w, h]
        
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        return [x_min, y_min, x_max - x_min, y_max - y_min]
    
    def _transform_bbox(self, bbox, mat, ori_size):
        """使用仿射变换矩阵变换bbox"""
        x, y, w, h = bbox
        # 四个角点
        corners = np.array([
            [x, y, 1],
            [x + w, y, 1],
            [x, y + h, 1],
            [x + w, y + h, 1]
        ], dtype=np.float32)
        
        # 应用仿射变换
        transformed = corners @ mat.T
        
        # 计算新的bbox
        x_min, y_min = transformed[:, 0].min(), transformed[:, 1].min()
        x_max, y_max = transformed[:, 0].max(), transformed[:, 1].max()
        
        return [x_min, y_min, x_max - x_min, y_max - y_min]
    
    def _normalize_bbox(self, bbox, img_size):
        """归一化bbox到0-1范围，并转换为 [cx, cy, w, h] 格式"""
        inp_h, inp_w = img_size
        x, y, w, h = bbox
        
        # 转换为中心点格式并归一化
        cx = (x + w / 2) / inp_w
        cy = (y + h / 2) / inp_h
        nw = w / inp_w
        nh = h / inp_h
        
        # clamp到合理范围
        cx = np.clip(cx, 0, 1)
        cy = np.clip(cy, 0, 1)
        nw = np.clip(nw, 0, 1)
        nh = np.clip(nh, 0, 1)
        
        return [cx, cy, nw, nh]

    # def get_length(self):
    #     return self.length

    # def get_sample(self, idx):
    #     return self.__getitem__(idx)
