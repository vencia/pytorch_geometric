import os.path as osp
import glob

import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.read import read_off


class Meshes(InMemoryDataset):
    def __init__(self,
                 root,
                 vertices=6,
                 hole=False,
                 train=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.vertices = vertices
        self.hole = hole
        super().__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['input', 'gt']

    @property
    def processed_file_names(self):
        return ['training_v{}{}.pt'.format(self.vertices, '_hole' if self.hole else ''),
                'test_v{}{}.pt'.format(self.vertices, '_hole' if self.hole else '')]

    def download(self):
        pass

    def process(self):
        data_list = []
        for ip_path in glob.glob('{}/*.off'.format(self.raw_paths[0])):
            ip_data = read_off(ip_path)
            shape_id = osp.basename(ip_path).rsplit('.', 1)[0]
            gt_path = osp.join(self.raw_paths[1], shape_id + '.off')
            gt_data = read_off(gt_path)

            data = Data()
            data.shape_id = torch.tensor([int(shape_id.split('_')[1]), int(shape_id.split('_')[2])])
            data.x = ip_data.pos
            data.gt_x = gt_data.pos
            data.face = ip_data.face
            data.gt_face = gt_data.face

            # if int(shape_id) < 10:
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        val_split = 0.15
        val_size = int(len(data_list) * val_split)

        torch.save(self.collate(data_list[:len(data_list) - val_size]), self.processed_paths[0])
        torch.save(self.collate(data_list[len(data_list) - val_size:]), self.processed_paths[1])
