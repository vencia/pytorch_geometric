import torch
from torch_geometric.utils import to_undirected


class FaceToEdge(object):
    r"""Converts mesh faces :obj:`[3, num_faces]` to edge indices
    :obj:`[2, num_edges]`.

    Args:
        remove_faces (bool, optional): If set to :obj:`False`, the face tensor
            will not be removed.
    """

    def __init__(self, remove_faces=True):
        self.remove_faces = remove_faces

    def __call__(self, data):
        face = data.face

        edge_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
        edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)

        data.edge_index = edge_index
        if self.remove_faces:
            data.face = None
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class FaceToEdgeWithLabels(FaceToEdge):

    def __call__(self, data):
        face = data.face

        edge_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
        edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)

        data.edge_index = edge_index
        edge_labels = torch.zeros([edge_index.shape[1]], dtype=torch.int32)
        for f, l in zip(face, data.y):
            edge_labels[f[0]] = l
            edge_labels[f[1]] = l
            edge_labels[f[2]] = l
        data.y = edge_labels
        if self.remove_faces:
            data.face = None
        return data
