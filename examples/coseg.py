import os.path as osp
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
from torch_geometric.datasets.coseg import COSEG
import torch_geometric.transforms as T
from torch_geometric.transforms.face_to_edge import FaceToEdgeWithLabels
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv

NUM_CLASSES = 4


class CalcEdgeAttributesTransform(object):
    def __call__(self, data):
        print('shape', data.shape_id.item())
        edges = data.edge_index.t().contiguous()
        faces = data.face.t()
        edge_attributes = np.empty([edges.shape[0], 5], dtype=np.float32)
        positions = data.pos
        for c, edge in enumerate(edges):
            incident_faces = [x for x in faces if edge[0] in x and edge[1] in x]
            assert 1 <= len(incident_faces) <= 2
            r = [x for x in incident_faces[0] if x != edge[0] and x != edge[1]][0]  # other point of face a
            s = None  # other point of face b
            if len(incident_faces) > 1:
                s = [x for x in incident_faces[1] if x != edge[0] and x != edge[1]][0]

            # calculate stable dihedral angle
            b1 = positions[r] - positions[edge[0]]
            b2 = positions[edge[1]] - positions[edge[0]]
            b2_norm = b2 / np.linalg.norm(b2)
            if s is not None:
                b3 = positions[s] - positions[edge[1]]
                n1 = np.cross(b1, b2)
                n1 /= np.linalg.norm(n1)
                n2 = np.cross(b2, b3)
                n2 /= np.linalg.norm(n2)
                m1 = np.cross(n1, b2_norm)
                dihedral_angle = np.abs(np.arctan2(np.dot(m1, n2), np.dot(n1, n2)))
            else:
                dihedral_angle = 0.0

            # calculate inner angles and edge ratios
            inner_angles = []
            edge_ratios = []
            for p in [r, s]:
                if p is not None:
                    v1 = positions[edge[0]] - positions[p]
                    v1_norm = v1 / np.linalg.norm(v1)
                    v2 = positions[edge[1]] - positions[p]
                    v2_norm = v2 / np.linalg.norm(v2)
                    inner_angle = np.arccos(np.dot(v1_norm, v2_norm))

                    assert 0 <= inner_angle <= 2 * np.pi
                    inner_angles.append(inner_angle)

                    perpendicular_vector = -v1 - (np.dot(-v1, b2_norm) * b2_norm)
                    edge_ratio = np.linalg.norm(b2) / np.linalg.norm(perpendicular_vector)
                    edge_ratios.append(edge_ratio)
                else:
                    inner_angles.append(0.0)
                    edge_ratios.append(0.0)

            inner_angles = sorted(inner_angles)
            edge_ratios = sorted(edge_ratios)
            edge_attributes[c] = [dihedral_angle, *inner_angles, *edge_ratios]
        data.edge_attr = torch.from_numpy(edge_attributes)
        return data


def main():
    data_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'coseg', 'vases')
    current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    run_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'runs', current_time)
    pre_transform = T.Compose(
        [T.NormalizeScale(), FaceToEdgeWithLabels(remove_faces=False), CalcEdgeAttributesTransform()])
    train_dataset = COSEG(data_path, train=True, pre_transform=pre_transform)
    test_dataset = COSEG(data_path, train=False, pre_transform=pre_transform)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = Net().to(device)

    train_writer = SummaryWriter(run_path + '/train')
    test_writer = SummaryWriter(run_path + '/test')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        train_acc, train_loss = train(model, device, optimizer, train_loader)
        train_writer.add_scalar('accuracy', train_acc, epoch)
        train_writer.add_scalar('loss', train_loss, epoch)
        test_acc, test_loss = test(model, device, test_loader)
        test_writer.add_scalar('accuracy', test_acc, epoch)
        test_writer.add_scalar('loss', test_loss, epoch)
        print('epoch: {:04d}, train acc: {:.4f}, test acc: {:.4f}'.format(epoch + 1, train_acc, test_acc))

    train_writer.close()
    test_writer.close()

    for data in train_loader:
        predict(model, device, data, run_path + '/train')
    for data in test_loader:
        predict(model, device, data, run_path + '/test')


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(5, 16)
        self.conv2 = GCNConv(16, 64)
        self.conv3 = GCNConv(64, NUM_CLASSES)

    def forward(self, data):
        edge_index, x = data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.log_softmax(self.conv3(x, edge_index), dim=-1)
        return x


def train(model, device, optimizer, train_loader):
    model.train()
    correct = 0
    loss = 0

    for data in train_loader:
        optimizer.zero_grad()
        pred = model(data.to(device))
        pred_class = pred.max(dim=-1)[1]
        correct += pred_class.eq(data.y).sum().item() / data.num_edges
        output = F.nll_loss(pred, data.y)
        loss += output.item()
        output.backward()
        optimizer.step()
    return correct / len(train_loader), loss / len(train_loader)


def test(model, device, test_loader):
    model.eval()
    correct = 0
    loss = 0

    for data in test_loader:
        pred = model(data.to(device))
        pred_class = pred.max(dim=-1)[1]
        correct += pred_class.eq(data.y).sum().item() / data.num_edges
        output = F.nll_loss(pred, data.y)
        loss += output.item()

    return correct / len(test_loader), loss / len(test_loader)


def predict(model, device, data, output_path):
    model.eval()
    pred = model(data.to(device))
    pred_class = pred.max(dim=-1)[1]
    export_colored_mesh(data, data.y, output_path + '/' + str(data.shape_id.item()) + '_gt.off')
    export_colored_mesh(data, pred_class, output_path + '/' + str(data.shape_id.item()) + '_pred.off')


def export_colored_mesh(data, labels, output_path):
    def get_color(label):
        if label == 0:
            return ['0.0', '0.0', '1.0', '1.0']
        elif label == 1:
            return ['0.0', '1.0', '0.0', '1.0']
        elif label == 2:
            return ['1.0', '0.0', '0.0', '1.0']
        else:
            assert label == 3
            return ['1.0', '1.0', '0.0', '1.0']

    vertices = data.pos.cpu().numpy()
    edges = data.edge_index.t().cpu().numpy()
    faces = data.face.t().cpu().numpy()
    edge_labels = labels.cpu().numpy()

    # edge labels to face labels
    face_labels = []
    for face in faces:
        l = edge_labels[np.nonzero(np.all(edges == [face[0], face[1]], axis=1))[0][0]]
        face_labels.append(l)

    # write off file
    lines = ['OFF']
    lines.append('{} {} {}'.format(len(vertices), len(faces), len(edges)))
    lines += [' '.join(str(y) for y in x) for x in vertices]
    lines += [' '.join(['3'] + [str(y) for y in x] + get_color(l)) for x, l in zip(faces, face_labels)]
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def export_labels(data, labels, path):
    pred_class = [str(x.item() + 1) for x in labels]
    with open(path + '/' + str(data.shape_id.item()) + '.seg', 'w') as f:
        f.write('\n'.join(pred_class))


if __name__ == '__main__':
    main()
