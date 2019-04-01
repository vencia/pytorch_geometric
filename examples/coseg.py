import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import COSEG
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import SplineConv
from torch_geometric.utils import degree


class CalcEdgeAttributesTransform(object):
    def __call__(self, data):
        edges = data.edge_index.t().contiguous()
        faces = data.face.t()
        edge_attributes = np.empty([edges.shape[0], 5])
        positions = data.pos
        for c, edge in enumerate(edges):
            print(c)
            incident_faces = [x for x in faces if edge[0] in x and edge[1] in x]
            assert len(incident_faces) == 2
            r = [x for x in incident_faces[0] if x != edge[0] and x != edge[1]][0]  # other point of face a
            s = [x for x in incident_faces[1] if x != edge[0] and x != edge[1]][0]  # other point of face b

            # calculate stable dihedral angle
            b1 = positions[r] - positions[edge[0]]
            b2 = positions[edge[1]] - positions[edge[0]]
            b2_norm = b2 / np.linalg.norm(b2)
            b3 = positions[s] - positions[edge[1]]
            n1 = np.cross(b1, b2)
            n1 /= np.linalg.norm(n1)
            n2 = np.cross(b2, b3)
            n2 /= np.linalg.norm(n2)
            m1 = np.cross(n1, b2_norm)
            dihedral_angle = np.abs(np.arctan2(np.dot(m1, n2), np.dot(n1, n2)))

            # calculate inner angles and edge ratios
            inner_angles = []
            edge_ratios = []
            for p in [r, s]:
                v1 = positions[edge[0]] - positions[p]
                v1_norm = v1 / np.linalg.norm(v1)
                v2 = positions[edge[1]] - positions[p]
                v2_norm = v2 / np.linalg.norm(v2)
                inner_angle = np.arccos(np.dot(v1_norm, v2_norm))
                assert 0 < inner_angle < 2 * np.pi
                inner_angles.append(inner_angle)

                perpendicular_vector = -v1 - (np.dot(-v1, b2_norm) * b2_norm)
                edge_ratio = np.linalg.norm(b2) / np.linalg.norm(perpendicular_vector)
                edge_ratios.append(edge_ratio)

            inner_angles = sorted(inner_angles)
            edge_ratios = sorted(edge_ratios)
            edge_attributes[c] = [dihedral_angle, *inner_angles, *edge_ratios]
        data.edge_attr = torch.from_numpy(edge_attributes)
        return data


def norm(x, edge_index):
    deg = degree(edge_index[0], x.size(0), x.dtype) + 1
    return x / deg.unsqueeze(-1)


path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'coseg', 'vases')
pre_transform = T.Compose(
    [T.NormalizeScale(), T.FaceToEdgeWithLabels(remove_faces=False), CalcEdgeAttributesTransform()])
train_dataset = COSEG(path, train=True, pre_transform=pre_transform)
# test_dataset = COSEG(path, train=False, pre_transform=pre_transform)
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=1)
# d = train_dataset[0]


# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = SplineConv(1, 32, dim=3, kernel_size=5, norm=False)
#         self.conv2 = SplineConv(32, 64, dim=3, kernel_size=5, norm=False)
#         self.fc1 = torch.nn.Linear(64, 256)
#         self.fc2 = torch.nn.Linear(256, d.num_nodes)
#
#     def forward(self, data):
#         x, edge_index, pseudo = data.x, data.edge_index, data.edge_attr
#         x = F.elu(norm(self.conv1(x, edge_index, pseudo), edge_index))
#         x = F.elu(norm(self.conv2(x, edge_index, pseudo), edge_index))
#         x = F.elu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)
#
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Net().to(device)
# # target = torch.arange(d.num_edges, dtype=torch.long, device=device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#
#
# def train(epoch):
#     model.train()
#
#     if epoch == 61:
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = 0.001
#
#     for data in train_loader:
#         optimizer.zero_grad()
#         F.nll_loss(model(data.to(device)), data.y).backward()
#         optimizer.step()
#
#
# def test():
#     model.eval()
#     correct = 0
#
#     for data in test_loader:
#         pred = model(data.to(device)).max(1)[1]
#         correct += pred.eq(data.y).sum().item()
#     return correct / (len(test_dataset) * d.num_edges)
#
#
# for epoch in range(1, 101):
#     train(epoch)
#     test_acc = test()
#     print('Epoch: {:02d}, Test: {:.4f}'.format(epoch, test_acc))
