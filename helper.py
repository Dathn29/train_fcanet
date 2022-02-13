import cv2
import numpy as np
from PIL import Image
from scipy.ndimage.morphology import distance_transform_edt
import random

random.seed(10)


########################################[ General ]########################################
def get_points_mask(size, points):
    mask = np.zeros(size[::-1]).astype(np.uint8)
    if len(points) != 0:
        points = np.array(points)
        mask[points[:, 1], points[:, 0]] = 1
    return mask


def get_points_list(mask):
    pts_y, pts_x = np.where(mask == 1)
    pts_xy = np.concatenate((pts_x[:, np.newaxis], pts_y[:, np.newaxis]), axis=1)
    return pts_xy.tolist()


def get_points_random(mask, point_num):
    if point_num == 0: return []
    pts_y, pts_x = np.where(mask == 1)
    pts_xy = np.concatenate((pts_x[:, np.newaxis], pts_y[:, np.newaxis]), axis=1)
    pts_num = len(pts_xy)
    if pts_num == 0: return []
    if point_num >= pts_num: point_num = 1
    indices = random.sample(range(pts_num), point_num)
    points = pts_xy[indices]
    return points


########################################[ Robot Strategy ]########################################

# pred 0-1 gt 0-1-255
def get_anno_point(pred, gt, anno_points):
    fn_map, fp_map = (gt == 1) & (pred == 0), (gt == 0) & (pred == 1)

    fn_map = np.pad(fn_map, ((1, 1), (1, 1)), 'constant')
    fndist_map = distance_transform_edt(fn_map)
    fndist_map = fndist_map[1:-1, 1:-1]

    fp_map = np.pad(fp_map, ((1, 1), (1, 1)), 'constant')
    fpdist_map = distance_transform_edt(fp_map)
    fpdist_map = fpdist_map[1:-1, 1:-1]

    if isinstance(anno_points, list):
        for pt in anno_points:
            fndist_map[pt[1], pt[0]] = fpdist_map[pt[1], pt[0]] = 0
    else:
        fndist_map[anno_points == 1] = 0
        fpdist_map[anno_points == 1] = 0

    if np.max(fndist_map) > np.max(fpdist_map):
        usr_map, if_pos = fndist_map, True
    else:
        usr_map, if_pos = fpdist_map, False

    [y_mlist, x_mlist] = np.where(usr_map == np.max(usr_map))
    pt_new = (x_mlist[0], y_mlist[0])
    return pt_new, if_pos


########################################[ Train Sample Strategy ]########################################

def get_pos_points_walk(gt, pos_point_num, step=0.2, margin=0.2):
    if pos_point_num == 0: return []

    pos_points = []
    choice_map_margin = (gt == 1).astype(np.int64)
    choice_map_margin = np.pad(choice_map_margin, ((1, 1), (1, 1)), 'constant')
    dist_map_margin = distance_transform_edt(choice_map_margin)[1:-1, 1:-1]

    if isinstance(margin, list):
        margin = random.choice(margin)

    if margin > 0 and margin < 1.0:
        margin = int(dist_map_margin.max() * margin)

    choice_map_margin = dist_map_margin > margin

    choice_map_step = np.ones_like(gt).astype(np.int64)
    choice_map_step = np.pad(choice_map_step, ((1, 1), (1, 1)), 'constant')

    if isinstance(step, list):
        step = random.choice(step)

    if step > 0 and step < 1.0:
        step = int(np.sqrt((gt == 1).sum() / np.pi) * 2 * step)

    for i in range(pos_point_num):
        dist_map_step = distance_transform_edt(choice_map_step)[1:-1, 1:-1]
        pts_y, pts_x = np.where((choice_map_margin) & (dist_map_step > step))
        pts_xy = np.concatenate((pts_x[:, np.newaxis], pts_y[:, np.newaxis]), axis=1)
        if len(pts_xy) == 0: break
        pt_new = tuple(pts_xy[random.randint(0, len(pts_xy) - 1), :])
        pos_points.append(pt_new)
        choice_map_step[pt_new[1] + 1, pt_new[0] + 1] = 0

    return pos_points


def get_neg_points_walk(gt, neg_point_num, margin_min=0.06, margin_max=0.48, step=0.2):
    if neg_point_num == 0: return []

    neg_points = []

    if isinstance(margin_min, list):
        margin_min = random.choice(margin_min)
    if isinstance(margin_max, list):
        margin_max = random.choice(margin_max)

    if margin_min > 0 and margin_min < 1.0 and margin_max > 0 and margin_max < 1.0:
        fg = (gt == 1).astype(np.int64)
        fg = np.pad(fg, ((1, 1), (1, 1)), 'constant')
        dist_fg = distance_transform_edt(fg)[1:-1, 1:-1]
        margin_max = min(max(int(dist_fg.max() * margin_min), 3), 10) * (margin_max / margin_min)
        margin_min = min(max(int(dist_fg.max() * margin_min), 3), 10)

    choice_map_margin = (gt != 1).astype(np.int64)
    dist_map_margin = distance_transform_edt(choice_map_margin)
    choice_map_margin = (dist_map_margin > margin_min) & (dist_map_margin < margin_max)

    choice_map_step = np.ones_like(gt).astype(np.int64)
    choice_map_step = np.pad(choice_map_step, ((1, 1), (1, 1)), 'constant')

    if isinstance(step, list):
        step = random.choice(step)

    if step > 0 and step < 1.0:
        step = int(np.sqrt((gt == 1).sum() / np.pi) * 2 * step)

    for i in range(neg_point_num):
        dist_map_step = distance_transform_edt(choice_map_step)[1:-1, 1:-1]
        pts_y, pts_x = np.where((choice_map_margin) & (dist_map_step > step))
        pts_xy = np.concatenate((pts_x[:, np.newaxis], pts_y[:, np.newaxis]), axis=1)
        if len(pts_xy) == 0: break
        pt_new = tuple(pts_xy[random.randint(0, len(pts_xy) - 1), :])
        neg_points.append(pt_new)
        choice_map_step[pt_new[1] + 1, pt_new[0] + 1] = 0

    return neg_points
