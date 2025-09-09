# coding=utf-8
import os
import cv2
import json
import re
import random
import spacy

nlp = spacy.load("en_core_web_lg")

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from utils1.utils import label2yolobox


class RefCOCODataSet(Dataset):
    def __init__(self, args, split):
        super(RefCOCODataSet, self).__init__()
        self.args = args  # 直接使用 args 字典
        self.split = split

        # 支持的数据集检查
        assert args['DATASET'] in ['refcoco', 'refcoco+', 'refcocog', 'referit'], f"Unknown dataset: {args['DATASET']}"

        # 读取标注文件
        ann_path = args['ANN_PATH'][args['DATASET']]
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")
        stat_refs_list = json.load(open(ann_path, 'r'))

        # 获取当前 split 的 refs
        splits = split.split('+')
        self.refs_anno = []
        for s in splits:
            if s in stat_refs_list:
                self.refs_anno.extend(stat_refs_list[s])
            else:
                raise ValueError(f"Split '{s}' not found in annotation file.")

        # 数据路径设置
        self.image_path = args['IMAGE_PATH'][args['DATASET']]
        self.mask_path = args['MASK_PATH'][args['DATASET']]
        self.input_shape = args['INPUT_SHAPE']  # [H, W]

        # 统计数据量
        self.data_size = len(self.refs_anno)
        print(f' ========== Dataset: {args["DATASET"]}, Split: {split}, Size: {self.data_size}')

        # 构建词表（使用所有数据构建，而不仅是当前 split）
        all_refs = []
        for key in stat_refs_list:
            for ann in stat_refs_list[key]:
                all_refs.extend(ann['refs'])

        # 构建 token 映射
        self.token_to_ix, self.ix_to_token, self.pretrained_emb, max_token = self.tokenize(all_refs, use_glove=True)
        self.token_size = len(self.token_to_ix)
        print(f' ========== Token vocab size: {self.token_size}')

        # 设置最大 token 长度
        self.max_token = args['MAX_TOKEN'] if args['MAX_TOKEN'] != -1 else max_token
        print(f'Max token length: {max_token} -> Trimmed to: {self.max_token}')

        # 图像变换
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(args['MEAN'], args['STD'])
        ])

    def tokenize(self, all_refs, use_glove=True):
        token_to_ix = {'PAD': 0, 'UNK': 1, 'CLS': 2}
        pretrained_emb = []

        if use_glove:
            # 使用 spaCy 的词向量
            pretrained_emb.append(nlp('PAD').vector)
            pretrained_emb.append(nlp('UNK').vector)
            pretrained_emb.append(nlp('CLS').vector)

        max_token = 0
        for ref in all_refs:
            words = re.sub(r"([.,'!?\"()*#:;])", '', ref.lower()).replace('-', ' ').replace('/', ' ').split()
            if len(words) > max_token:
                max_token = len(words)

            for word in words:
                if word not in token_to_ix:
                    token_to_ix[word] = len(token_to_ix)
                    if use_glove:
                        pretrained_emb.append(nlp(word).vector)

        pretrained_emb = np.array(pretrained_emb) if use_glove else None
        ix_to_token = {ix: word for word, ix in token_to_ix.items()}
        return token_to_ix, ix_to_token, pretrained_emb, max_token

    def proc_ref(self, ref):
        """将文本转换为 index 序列"""
        ques_ix = np.zeros(self.max_token, dtype=np.int64)
        words = re.sub(r"([.,'!?\"()*#:;])", '', ref.lower()).replace('-', ' ').replace('/', ' ').split()

        for i, word in enumerate(words):
            if i >= self.max_token:
                break
            ques_ix[i] = self.token_to_ix.get(word, self.token_to_ix['UNK'])
        return ques_ix

    def load_refs(self, idx):
        """随机选择一个 ref 表达"""
        refs = self.refs_anno[idx]['refs']
        ref_txt = random.choice(refs)
        ref = self.proc_ref(ref_txt)
        return ref, ref_txt

    def preprocess_info(self, img, mask, box, iid, lr_flip=False):
        """图像、mask、box 预处理为固定尺寸"""
        h, w = img.shape[:2]
        imgsize = self.input_shape[0]
        ar = w / h
        if ar < 1:
            nh, nw = imgsize, int(imgsize * ar)
        else:
            nw, nh = imgsize, int(imgsize / ar)
        dx = (imgsize - nw) // 2
        dy = (imgsize - nh) // 2

        # 调整图像
        resized_img = cv2.resize(img, (nw, nh))
        sized = np.ones((imgsize, imgsize, 3), dtype=np.uint8) * 127
        sized[dy:dy + nh, dx:dx + nw, :] = resized_img

        info_img = (h, w, nh, nw, dx, dy, iid)

        mask = np.expand_dims(mask, -1).astype(np.float32)
        mask = cv2.resize(mask, (nw, nh))
        mask = np.expand_dims(mask, -1).astype(np.float32)
        sized_mask = np.zeros((imgsize, imgsize, 1), dtype=np.float32)
        sized_mask[dy:dy + nh, dx:dx + nw, :] = mask
        sized_mask = np.transpose(sized_mask, (2, 0, 1))

        sized_box = label2yolobox(box, info_img, self.input_shape[0], lrflip=lr_flip)
        return sized, sized_mask, sized_box, info_img

    def load_img_feats(self, idx):
        """加载图像特征"""
        if self.args['DATASET'] in ['refcoco', 'refcoco+', 'refcocog']:
            img_path = os.path.join(self.image_path, f'COCO_train2014_{self.refs_anno[idx]["iid"]:012d}.jpg')
        elif self.args['DATASET'] == 'referit':
            img_path = os.path.join(self.image_path, f'{self.refs_anno[idx]["iid"]}.jpg')
        else:
            raise NotImplementedError("Unsupported dataset")

        image = cv2.imread(img_path)
        if self.args['DATASET'] in ['refcoco', 'refcoco+', 'refcocog', 'referit']:
            mask = np.load(os.path.join(self.mask_path, f'{self.refs_anno[idx]["mask_id"]}.npy'))
        else:
            mask = np.zeros([image.shape[0], image.shape[1], 1], dtype=np.float32)

        box = np.array([self.refs_anno[idx]['bbox']])
        return image, mask, box, self.refs_anno[idx]['mask_id'], self.refs_anno[idx]['iid'], img_path

    def __getitem__(self, idx):
        ref_iter, ref_txt = self.load_refs(idx)
        image_iter, mask_iter, gt_box_iter, mask_id, iid, img_path = self.load_img_feats(idx)
        image_iter_ori = cv2.cvtColor(image_iter, cv2.COLOR_BGR2RGB)
        flip_box = False
        image_iter, mask_iter, box_iter, info_iter = self.preprocess_info(image_iter_ori, mask_iter, gt_box_iter.copy(),
                                                                          iid, flip_box)
        image_iter = self.transforms(image_iter)
        return \
            torch.from_numpy(ref_iter).long(), \
            image_iter, \
            torch.from_numpy(mask_iter).float(), \
            torch.from_numpy(box_iter).float(), \
            torch.from_numpy(gt_box_iter).float(), \
            mask_id, \
            np.array(info_iter), \
            ref_txt, \
            img_path

    def __len__(self):
        return self.data_size


def loader(args, dataset: Dataset, rank: int, shuffle, drop_last=False):
    data_loader = DataLoader(
        dataset,
        batch_size=args['Batch_size'],
        shuffle=shuffle,
        num_workers=args['nW'],
        pin_memory=False,
        drop_last=drop_last
    )
    return data_loader


# # coding=utf-8
#
# import os
# import cv2
# # import json, re, en_vectors_web_lg, random
# import json, re, random
# import spacy
# nlp = spacy.load("en_core_web_lg")  # 替代 en_vectors_web_lg
#
# import numpy as np
#
# import torch
# import torch.utils.data as Data
# from torch.utils.data import DataLoader
# from torchvision.transforms import transforms
#
# from utils1.utils import label2yolobox
#
#
# class RefCOCODataSet(Data.Dataset):
#     def __init__(self, __C, split):
#         super(RefCOCODataSet, self).__init__()
#         self.__C = __C
#         self.split = split
#         assert __C.DATASET in ['refcoco', 'refcoco+', 'refcocog', 'referit']
#         stat_refs_list = json.load(open(__C.ANN_PATH[__C.DATASET], 'r'))
#         total_refs_list = []
#         self.ques_list = []
#         splits = split.split('+')
#         self.refs_anno = []
#         for split_ in splits:
#             self.refs_anno += stat_refs_list[split_]
#         refs = []
#         for split in stat_refs_list:
#             for ann in stat_refs_list[split]:
#                 for ref in ann['refs']:
#                     refs.append(ref)
#         for split in total_refs_list:
#             for ann in total_refs_list[split]:
#                 for ref in ann['refs']:
#                     refs.append(ref)
#
#         self.image_path = __C.IMAGE_PATH[__C.DATASET]
#         self.mask_path = __C.MASK_PATH[__C.DATASET]
#         self.input_shape = __C.INPUT_SHAPE
#         # Define run data size
#         self.data_size = len(self.refs_anno)
#
#         print(' ========== Dataset size:', self.data_size)
#         self.token_to_ix, self.ix_to_token, self.pretrained_emb, max_token = self.tokenize(stat_refs_list,
#                                                                                            __C.USE_GLOVE)
#         self.token_size = self.token_to_ix.__len__()
#         print(' ========== Question token vocab size:', self.token_size)
#
#         self.max_token = __C.MAX_TOKEN
#         if self.max_token == -1:
#             self.max_token = max_token
#         print('Max token length:', max_token, 'Trimmed to:', self.max_token)
#         print('Finished!')
#         print('')
#
#         self.candidate_transforms = {}
#         self.transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(__C.MEAN, __C.STD)])
#
#     def tokenize(self, stat_refs_list, use_glove):
#         token_to_ix = {
#             'PAD': 0,
#             'UNK': 1,
#             'CLS': 2,
#         }
#
#         spacy_tool = None
#         pretrained_emb = []
#         if use_glove:
#             spacy_tool = en_vectors_web_lg.load()
#             pretrained_emb.append(spacy_tool('PAD').vector)
#             pretrained_emb.append(spacy_tool('UNK').vector)
#             pretrained_emb.append(spacy_tool('CLS').vector)
#
#         max_token = 0
#         for split in stat_refs_list:
#             for ann in stat_refs_list[split]:
#                 for ref in ann['refs']:
#                     words = re.sub(
#                         r"([.,'!?\"()*#:;])",
#                         '',
#                         ref.lower()
#                     ).replace('-', ' ').replace('/', ' ').split()
#
#                     if len(words) > max_token:
#                         max_token = len(words)
#
#                     for word in words:
#                         if word not in token_to_ix:
#                             token_to_ix[word] = len(token_to_ix)
#                             if use_glove:
#                                 pretrained_emb.append(spacy_tool(word).vector)
#
#         pretrained_emb = np.array(pretrained_emb)
#         ix_to_token = {}
#         for item in token_to_ix:
#             ix_to_token[token_to_ix[item]] = item
#
#         return token_to_ix, ix_to_token, pretrained_emb, max_token
#
#     def proc_ref(self, ref, token_to_ix, max_token):
#         ques_ix = np.zeros(max_token, np.int64)
#
#         words = re.sub(
#             r"([.,'!?\"()*#:;])",
#             '',
#             ref.lower()
#         ).replace('-', ' ').replace('/', ' ').split()
#
#         for ix, word in enumerate(words):
#             if word in token_to_ix:
#                 ques_ix[ix] = token_to_ix[word]
#             else:
#                 ques_ix[ix] = token_to_ix['UNK']
#
#             if ix + 1 == max_token:
#                 break
#
#         return ques_ix
#
#     def load_refs(self, idx):
#         refs = self.refs_anno[idx]['refs']
#         ref_txt = refs[np.random.choice(len(refs))]
#         ref = self.proc_ref(ref_txt, self.token_to_ix, self.max_token)
#         return ref, ref_txt
#
#     def preprocess_info(self, img, mask, box, iid, lr_flip=False):
#         h, w, _ = img.shape
#         imgsize = self.input_shape[0]
#         new_ar = w / h
#         if new_ar < 1:
#             nh = imgsize
#             nw = nh * new_ar
#         else:
#             nw = imgsize
#             nh = nw / new_ar
#         nw, nh = int(nw), int(nh)
#         dx = (imgsize - nw) // 2
#         dy = (imgsize - nh) // 2
#         img = cv2.resize(img, (nw, nh))
#         sized = np.ones((imgsize, imgsize, 3), dtype=np.uint8) * 127
#         sized[dy:dy + nh, dx:dx + nw, :] = img
#         info_img = (h, w, nh, nw, dx, dy, iid)
#
#         mask = np.expand_dims(mask, -1).astype(np.float32)
#         mask = cv2.resize(mask, (nw, nh))
#         mask = np.expand_dims(mask, -1).astype(np.float32)
#         sized_mask = np.zeros((imgsize, imgsize, 1), dtype=np.float32)
#         sized_mask[dy:dy + nh, dx:dx + nw, :] = mask
#         sized_mask = np.transpose(sized_mask, (2, 0, 1))
#         sized_box = label2yolobox(box, info_img, self.input_shape[0], lrflip=lr_flip)
#         return sized, sized_mask, sized_box, info_img
#
#     def load_img_feats(self, idx):
#         img_path = None
#         if self.__C.DATASET in ['refcoco', 'refcoco+', 'refcocog']:
#             img_path = os.path.join(self.image_path, 'COCO_train2014_%012d.jpg' % self.refs_anno[idx]['iid'])
#         elif self.__C.DATASET == 'referit':
#             img_path = os.path.join(self.image_path, '%d.jpg' % self.refs_anno[idx]['iid'])
#         else:
#             assert NotImplementedError
#         image = cv2.imread(img_path)
#         if self.__C.DATASET in ['refcoco', 'refcoco+', 'refcocog', 'referit']:
#             mask = np.load(os.path.join(self.mask_path, '%d.npy' % self.refs_anno[idx]['mask_id']))
#         else:
#             mask = np.zeros([image.shape[0], image.shape[1], 1], dtype=np.float)
#
#         box = np.array([self.refs_anno[idx]['bbox']])
#         return image, mask, box, self.refs_anno[idx]['mask_id'], self.refs_anno[idx]['iid'], img_path
#
#     def __getitem__(self, idx):
#         ref_iter, ref_txt = self.load_refs(idx)
#         image_iter, mask_iter, gt_box_iter, mask_id, iid, img_path = self.load_img_feats(idx)
#         image_iter_ori = cv2.cvtColor(image_iter, cv2.COLOR_BGR2RGB)
#         ops = None
#         if len(list(self.candidate_transforms.keys())) > 0:
#             ops = random.choices(list(self.candidate_transforms.keys()), k=1)[0]
#         if ops is not None and ops != 'RandomErasing':
#             image_iter = self.candidate_transforms[ops](image=image_iter_ori)['image']
#         flip_box = False
#         image_iter, mask_iter, box_iter, info_iter = self.preprocess_info(image_iter_ori, mask_iter, gt_box_iter.copy(),
#                                                                           iid,
#                                                                           flip_box)
#         image_iter = self.transforms(image_iter)
#         return \
#             torch.from_numpy(ref_iter).long(), \
#             image_iter, \
#             torch.from_numpy(mask_iter).float(), \
#             torch.from_numpy(box_iter).float(), \
#             torch.from_numpy(gt_box_iter).float(), \
#             mask_id, \
#             np.array(info_iter), \
#             ref_txt, \
#             img_path
#
#     def __len__(self):
#         return self.data_size
#
#     def shuffle_list(self, list):
#         random.shuffle(list)
#
#
# def loader(__C, dataset: torch.utils.data.Dataset, rank: int, shuffle, drop_last=False):
#     data_loader = DataLoader(dataset,
#                              batch_size=__C.BATCH_SIZE,
#                              shuffle=shuffle,
#                              num_workers=__C.NUM_WORKER,
#                              pin_memory=False,
#                              drop_last=drop_last)
#     return data_loader