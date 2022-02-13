import random
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset


class GeneralDataset(Dataset):
    def __init__(self, data_lists=[], transform=None):
        super().__init__()
        if not isinstance(data_lists, list):  data_lists = [data_lists]
        self.imgs_list, self.gts_list = [], []
        for data_list in data_lists:
            dataset_path = data_list.split('list')[0]
            img_suffix_ref = {file.stem: file.suffix for file in (Path(dataset_path) / 'img').glob('*.*')}
            gt_suffix_ref = {file.stem: file.suffix for file in (Path(dataset_path) / 'gt').glob('*.*')}
            with open(data_list) as f:
                ids = f.read().splitlines()
                for id in ids:
                    self.imgs_list.append(
                        str(Path(dataset_path) / 'img' / (id.split('#')[0] + img_suffix_ref[id.split('#')[0]])))
                    self.gts_list.append(str(Path(dataset_path) / 'gt' / (id + gt_suffix_ref[id])))
        self.transform = transform
    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, index):
        return self.transform(self.get_sample(index)) if self.transform != None else self.get_sample(index)

    def get_sample(self, index):
        img, gt = np.array(Image.open(self.imgs_list[index])), np.array(Image.open(self.gts_list[index]))
        gt = (gt == 1).astype(np.uint8) * 255

        sample = {'img': img, 'gt': gt}

        sample['meta'] = {'id': self.gts_list[index].rstrip('.png').split('/')[-1]}
        sample['meta']['source_size'] = np.array(gt.shape[::-1])
        sample['meta']['img_path'] = self.imgs_list[index]
        sample['meta']['gt_path'] = self.gts_list[index]
        return sample


