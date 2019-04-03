import os.path as osp
import glob

import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.read import read_off, read_txt_array


class COSEG(InMemoryDataset):
    urls = {
        'vases':
            ('http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Large-Vases/shapes.zip',
             'http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/data/Large-Vases/gt.zip')
    }

    def __init__(self,
                 root,
                 name='vases',
                 train=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        assert name in ['vases']
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['shapes', 'gt']

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def download(self):
        off_path = download_url(self.urls[self.name][0], self.root)
        label_path = download_url(self.urls[self.name][1], self.root)
        extract_zip(off_path, self.raw_dir)
        extract_zip(label_path, self.raw_dir)

    def process(self):
        data_list = []
        for off_path in glob.glob('{}/*.off'.format(self.raw_paths[0])):
            data = read_off(off_path)
            shape_id = osp.basename(off_path).rsplit('.', 1)[0]
            label_path = osp.join(self.raw_paths[1], shape_id + '.seg')
            data.y = read_txt_array(label_path)
            data.shape_id = torch.tensor([int(shape_id)])
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        torch.save(self.collate(data_list[:2]), self.processed_paths[0])  # TODO: change back
        torch.save(self.collate(data_list[2:]), self.processed_paths[1])
