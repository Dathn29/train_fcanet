import cv2
import math
import torch
import random
import numbers
import numpy as np
from PIL import Image

import helper
from scipy.ndimage.morphology import distance_transform_edt

MAX_VALUES_BY_DTYPE = {
    np.dtype("uint8"): 255,
    np.dtype("uint16"): 65535,
    np.dtype("uint32"): 4294967295,
    np.dtype("float32"): 1.0,
}
def _shift_image_uint8(img, value):
    max_value = MAX_VALUES_BY_DTYPE[img.dtype]

    lut = np.arange(0, max_value + 1).astype("float32")
    lut += value

    lut = np.clip(lut, 0, max_value).astype(img.dtype)
    return cv2.LUT(img, lut)
def _shift_rgb_uint8(img, r_shift, g_shift, b_shift):
    if r_shift == g_shift == b_shift:
        h, w, c = img.shape
        img = img.reshape([h, w * c])

        return _shift_image_uint8(img, r_shift)

    result_img = np.empty_like(img)
    shifts = [r_shift, g_shift, b_shift]
    for i, shift in enumerate(shifts):
        result_img[..., i] = _shift_image_uint8(img[..., i], shift)

    return result_img
def _shift_rgb_non_uint8(img, r_shift, g_shift, b_shift):
    if r_shift == g_shift == b_shift:
        return img + r_shift

    result_img = np.empty_like(img)
    shifts = [r_shift, g_shift, b_shift]
    for i, shift in enumerate(shifts):
        result_img[..., i] = img[..., i] + shift

    return result_img
class RGBShift(object):
    def __init__(self,r_shift=0,g_shift=0,b_shift=0):
        self.r_shift =r_shift
        self.g_shift = g_shift
        self.b_shift = b_shift
    def __call__(self, sample):
        img = sample['img']
        if img.dtype == np.uint8:
            sample['img']= _shift_rgb_uint8(img, self.r_shift, self.g_shift, self.b_shift)
        else:
            sample['img']=  _shift_rgb_non_uint8(img, self.r_shift, self.g_shift, self.b_shift)
        return sample

def img_resize_point(img, size):
    (h, w) = img.shape
    if not isinstance(size, tuple): size = (int(w * size), int(h * size))
    M = np.array([[size[0] / w, 0, 0], [0, size[1] / h, 0]])

    pts_y, pts_x = np.where(img == 1)
    pts_xy = np.concatenate((pts_x[:, np.newaxis], pts_y[:, np.newaxis]), axis=1)
    pts_xy_new = np.dot(np.insert(pts_xy, 2, 1, axis=1), M.T).astype(np.int64)

    img_new = np.zeros(size[::-1], dtype=np.uint8)
    for pt in pts_xy_new:
        img_new[pt[1], pt[0]] = 1
    return img_new


########################################[ General ]########################################
class ToTensor(object):
    def __init__(self, if_div=True, elems_do=None, elems_undo=[]):
        self.if_div = if_div
        self.elems_do, self.elems_undo = elems_do, (['meta'] + elems_undo)

    def __call__(self, sample):
        for elem in sample.keys():
            if self.elems_do != None and elem not in self.elems_do: continue
            if elem in self.elems_undo: continue
            tmp = sample[elem]
            tmp = tmp[np.newaxis, :, :] if tmp.ndim == 2 else tmp.transpose((2, 0, 1))
            tmp = torch.from_numpy(tmp).float()
            tmp = tmp.float().div(255) if self.if_div else tmp
            sample[elem] = tmp
        return sample


########################################[ Basic Image Augmentation ]########################################

class RandomFlip(object):
    def __init__(self, direction=Image.FLIP_LEFT_RIGHT, p=0.5, elems_do=None, elems_undo=[]):
        self.direction, self.p = direction, p
        self.elems_do, self.elems_undo = elems_do, (['meta'] + elems_undo)

    def __call__(self, sample):
        if random.random() < self.p:
            for elem in sample.keys():
                if self.elems_do != None and elem not in self.elems_do: continue
                if elem in self.elems_undo: continue
                sample[elem] = np.array(Image.fromarray(sample[elem]).transpose(self.direction))
            sample['meta']['flip'] = 1
        else:
            sample['meta']['flip'] = 0
        return sample


class Resize(object):
    def __init__(self, size, mode=None, elems_point=['pos_points_mask', 'neg_points_mask'], elems_do=None,
                 elems_undo=[]):
        self.size, self.mode = size, mode
        self.elems_point = elems_point
        self.elems_do, self.elems_undo = elems_do, (['meta'] + elems_undo)

    def __call__(self, sample):
        for elem in sample.keys():
            if self.elems_do != None and elem not in self.elems_do: continue
            if elem in self.elems_undo: continue

            if elem in self.elems_point:
                sample[elem] = img_resize_point(sample[elem], self.size)
                continue

            if self.mode is None:
                mode = cv2.INTER_LINEAR if len(sample[elem].shape) == 3 else cv2.INTER_NEAREST
            sample[elem] = cv2.resize(sample[elem], self.size, interpolation=mode)

        return sample


class Crop(object):
    def __init__(self, x_range, y_range, elems_do=None, elems_undo=[]):
        self.x_range, self.y_range = x_range, y_range
        self.elems_do, self.elems_undo = elems_do, (['meta'] + elems_undo)

    def __call__(self, sample):
        for elem in sample.keys():
            if self.elems_do != None and elem not in self.elems_do: continue
            if elem in self.elems_undo: continue
            sample[elem] = sample[elem][self.y_range[0]:self.y_range[1], self.x_range[0]:self.x_range[1], ...]

        sample['meta']['crop_size'] = np.array((self.x_range[1] - self.x_range[0], self.y_range[1] - self.y_range[0]))
        sample['meta']['crop_lt'] = np.array((self.x_range[0], self.y_range[0]))
        return sample


########################################[ Interactive Segmentation ]########################################

class MatchShortSideResize(object):
    def __init__(self, size, if_must=False, elems_do=None, elems_undo=[]):
        self.size, self.if_must = size, if_must
        self.elems_do, self.elems_undo = elems_do, (['meta'] + elems_undo)

    def __call__(self, sample):
        src_size = sample['gt'].shape[::-1]

        if self.if_must == False and src_size[0] >= self.size and src_size[1] >= self.size:
            return sample

        src_short_size = min(src_size[0], src_size[1])
        dst_size = (int(self.size * src_size[0] / src_short_size), int(self.size * src_size[1] / src_short_size))
        assert (dst_size[0] == self.size or dst_size[1] == self.size)
        Resize(size=dst_size)(sample)
        return sample


class FgContainCrop(object):
    def __init__(self, crop_size=(384, 384), if_whole=False, elems_do=None, elems_undo=[]):
        self.crop_size, self.if_whole = crop_size, if_whole
        self.elems_do, self.elems_undo = elems_do, (['meta'] + elems_undo)

    def __call__(self, sample):
        gt = sample['gt']
        src_size = gt.shape[::-1]
        x_range, y_range = [0, src_size[0] - self.crop_size[0]], [0, src_size[1] - self.crop_size[1]]
        if (gt > 127).any() == False:
            pass
        elif self.if_whole:
            bbox = cv2.boundingRect((gt > 127).astype(np.uint8))

            if bbox[2] <= self.crop_size[0]:
                x_range[1] = min(x_range[1], bbox[0])
                x_range[0] = max(x_range[0], bbox[0] + bbox[2] - self.crop_size[0])
            else:
                x_range = [bbox[0], bbox[0] + bbox[2] - self.crop_size[0]]

            if bbox[3] <= self.crop_size[1]:
                y_range[1] = min(y_range[1], bbox[1])
                y_range[0] = max(y_range[0], bbox[1] + bbox[3] - self.crop_size[1])
            else:
                y_range = [bbox[1], bbox[1] + bbox[3] - self.crop_size[1]]
        else:
            pts_y, pts_x = np.where(gt > 127)
            pts_xy = np.concatenate((pts_x[:, np.newaxis], pts_y[:, np.newaxis]), axis=1)
            sp_x, sp_y = pts_xy[random.randint(0, len(pts_xy) - 1)]
            x_range[1], y_range[1] = min(x_range[1], sp_x), min(y_range[1], sp_y)
            x_range[0], y_range[0] = max(x_range[0], sp_x + 1 - self.crop_size[0]), max(y_range[0],
                                                                                        sp_y + 1 - self.crop_size[1])

        x_st = random.randint(x_range[0], x_range[1])
        y_st = random.randint(y_range[0], y_range[1])
        Crop(x_range=(x_st, x_st + self.crop_size[0]), y_range=(y_st, y_st + self.crop_size[1]))(sample)
        return sample


########################################[ Interactive Segmentation (Points) ]########################################

class CatPointMask(object):
    def __init__(self, mode='NO', paras={}, if_repair=True):
        self.mode, self.paras, self.if_repair = mode, paras, if_repair

    def __call__(self, sample):
        gt = sample['gt']

        if 'pos_points_mask' in sample.keys() and self.if_repair:
            sample['pos_points_mask'][gt <= 127] = 0
        if 'neg_points_mask' in sample.keys() and self.if_repair:
            sample['neg_points_mask'][gt > 127] = 0

        if_gt_empty = not ((gt > 127).any())

        if (if_gt_empty == False) and (sample['pos_points_mask'].any() == False) and self.if_repair:
            if gt[gt.shape[0] // 2, gt.shape[1] // 2] > 127:
                sample['pos_points_mask'][gt.shape[0] // 2, gt.shape[1] // 2] = 1
            else:
                pts_y, pts_x = np.where(gt > 127)
                pts_xy = np.concatenate((pts_x[:, np.newaxis], pts_y[:, np.newaxis]), axis=1)
                pt_pos = pts_xy[random.randint(0, len(pts_xy) - 1)]
                sample['pos_points_mask'][pt_pos[1], pt_pos[0]] = 1

        pos_points_mask, neg_points_mask = sample['pos_points_mask'], sample['neg_points_mask']

        if self.mode == 'DISTANCE_POINT_MASK_SRC':
            max_dist = 255
            if if_gt_empty:
                pos_points_mask_dist = np.ones(gt.shape).astype(np.float64) * max_dist
            else:
                pos_points_mask_dist = distance_transform_edt(1 - pos_points_mask)
                pos_points_mask_dist = np.minimum(pos_points_mask_dist, max_dist)

            if neg_points_mask.any() == False:
                neg_points_mask_dist = np.ones(gt.shape).astype(np.float64) * max_dist
            else:
                neg_points_mask_dist = distance_transform_edt(1 - neg_points_mask)
                neg_points_mask_dist = np.minimum(neg_points_mask_dist, max_dist)

            pos_points_mask_dist, neg_points_mask_dist = pos_points_mask_dist * 255, neg_points_mask_dist * 255
            sample['pos_mask_dist_src'] = pos_points_mask_dist
            sample['neg_mask_dist_src'] = neg_points_mask_dist

        elif self.mode == 'DISTANCE_POINT_MASK_FIRST':
            max_dist = 255
            if if_gt_empty:
                pos_points_mask_dist = np.ones(gt.shape).astype(np.float64) * max_dist
            else:
                gt_tmp = (sample['gt'] > 127).astype(np.uint8)
                pred = np.zeros_like(gt_tmp)
                pt, if_pos = helper.get_anno_point(pred, gt_tmp, [])
                pos_points_mask = np.zeros_like(gt_tmp)
                pos_points_mask[pt[1], pt[0]] = 1
                pos_points_mask_dist = distance_transform_edt(1 - pos_points_mask)
                pos_points_mask_dist = np.minimum(pos_points_mask_dist, max_dist)
                pos_points_mask_dist = pos_points_mask_dist * 255
            sample['pos_mask_dist_first'] = pos_points_mask_dist

        return sample


class SimulatePoints(object):
    def __init__(self, mode='random', max_point_num=10, if_fixed=False):
        self.mode = mode
        self.max_point_num = max_point_num
        self.if_fixed = if_fixed

    def __call__(self, sample):
        if self.if_fixed:
            id = sample['meta']['id']
            str_seed = 0
            for c in id:
                str_seed += ord(c)
            str_seed = str_seed % 50
            random.seed(str_seed)

        gt = (sample['gt'] > 127).astype(np.uint8)

        if self.mode == 'random':
            pos_point_num, neg_point_num = random.randint(1, 10), random.randint(0, 10)

            pos_mask = (gt == 1)
            pos_points = helper.get_points_random(pos_mask, pos_point_num)

            bg_dist = distance_transform_edt(gt == 0)
            neg_mask = (bg_dist > 5) & (bg_dist < 40)
            neg_points = helper.get_points_random(neg_mask, neg_point_num)

            pos_points_mask, neg_points_mask = np.zeros_like(sample['gt']), np.zeros_like(sample['gt'])
            if len(pos_points) != 0:
                pos_points_mask[pos_points[:, 1], pos_points[:, 0]] = 1
            if len(neg_points) != 0:
                neg_points_mask[neg_points[:, 1], neg_points[:, 0]] = 1

            sample['pos_points_mask'] = pos_points_mask
            sample['neg_points_mask'] = neg_points_mask

        elif self.mode == 'strategy#05':
            pos_point_num, neg_point_num = random.randint(1, 10), random.randint(0, 10)

            gt = (sample['gt'] > 127).astype(np.uint8)
            pos_points = np.array(
                helper.get_pos_points_walk(gt, pos_point_num, step=[7, 10, 20], margin=[5, 10, 15, 20]))
            neg_points = np.array(
                helper.get_neg_points_walk(gt, neg_point_num, margin_min=[15, 40, 60], margin_max=[80],
                                            step=[10, 15, 25]))

            pos_points_mask, neg_points_mask = np.zeros_like(sample['gt']), np.zeros_like(sample['gt'])
            if len(pos_points) != 0:
                pos_points_mask[pos_points[:, 1], pos_points[:, 0]] = 1
            if len(neg_points) != 0:
                neg_points_mask[neg_points[:, 1], neg_points[:, 0]] = 1

            sample['pos_points_mask'] = pos_points_mask
            sample['neg_points_mask'] = neg_points_mask

        return sample


current_epoch = 0
record_anno = {}

record_crop_lt = {}
record_if_flip = {}


# ITIS
class ITIS_Crop(object):
    def __init__(self, itis_pro=0, mode='random', crop_size=(384, 384)):
        self.itis_pro = itis_pro
        self.mode = mode
        self.crop_size = crop_size

    def __call__(self, sample):
        global current_epoch, record_anno, record_crop_lt, record_if_flip
        id = sample['meta']['id']
        if (random.random() < self.itis_pro) and current_epoch != 0:
            Crop(x_range=(record_crop_lt[id][0], record_crop_lt[id][0] + self.crop_size[0]),
                 y_range=(record_crop_lt[id][1], record_crop_lt[id][1] + self.crop_size[1]))(sample)
            RandomFlip(p=(1.5 if record_if_flip[id] == 1 else -1))(sample)
            sample['pos_points_mask'] = helper.get_points_mask(sample['gt'].shape[::-1], record_anno[id][0])
            sample['neg_points_mask'] = helper.get_points_mask(sample['gt'].shape[::-1], record_anno[id][1])
        else:
            FgContainCrop(crop_size=self.crop_size, if_whole=False)(sample)
            RandomFlip(p=-1)(sample)
            SimulatePoints(mode=self.mode)(sample)

        return sample
