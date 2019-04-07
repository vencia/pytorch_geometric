import os
import os.path as osp
import numpy as np
from datetime import datetime
import click
from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
from torch_geometric.datasets.coseg import COSEG
import torch_geometric.transforms as T
from torch_geometric.transforms.face_to_edge import FaceToEdge
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn.pool.mesh_pool import mesh_pool


class CalcEdgeAttributesTransform(object):
    def __call__(self, data):
        print('shape', data.shape_id.item())
        edges = data.edge_index.t().contiguous()
        faces = data.face.t()
        edge_attributes = np.empty([len(edges), 5], dtype=np.float32)
        positions = data.pos
        vertex_degrees = np.zeros([len(positions)], dtype=int)
        for c, edge in enumerate(edges):
            vertex_degrees[edge[0]] += 1
            vertex_degrees[edge[1]] += 1
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

        # make edge attributes to node attributes by averaging over all incident edges
        vertex_attributes = np.zeros([positions.shape[0], 5], dtype=np.float32)
        for edge, edge_attr in zip(edges, edge_attributes):
            vertex_attributes[edge[0]] += edge_attr / vertex_degrees[edge[0]]
            vertex_attributes[edge[1]] += edge_attr / vertex_degrees[edge[1]]
        data.x = torch.from_numpy(vertex_attributes)
        return data


@click.command()
@click.option('--epochs', default=10)
@click.option('--lr', default=0.001)
@click.option('--classification', default=1)
@click.option('--pool', nargs=4, default=(5, 5, 5, 5))
def main(epochs, lr, classification, pool):
    print('pool', pool)
    data_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'coseg', 'vases')
    current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    run_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'runs', current_time)
    os.makedirs(run_path)
    with open(run_path + '/arguments.txt', 'w') as f:
        f.write('epochs {}\n'.format(epochs))
        f.write('lr {}\n'.format(lr))
        f.write('classification {}\n'.format(classification))
        f.write('pool {}\n'.format(pool))

    pre_transform = T.Compose([T.NormalizeScale(), FaceToEdge(remove_faces=False), CalcEdgeAttributesTransform()])
    train_dataset = COSEG(data_path, classification=classification, train=True, pre_transform=pre_transform)
    test_dataset = COSEG(data_path, classification=classification, train=False, pre_transform=pre_transform)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = Net(train_dataset.num_classes, pool).to(device)

    train_writer = SummaryWriter(run_path + '/train')
    test_writer = SummaryWriter(run_path + '/test')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for c, data in enumerate(train_loader):
        if data.shape_id.cpu().item() in [0, 1, 2]:
            export_colored_mesh(data, color=0, output_path=run_path + '/train/' + str(data.shape_id.item()) + '_gt.off')

    for epoch in range(epochs):
        train_acc, train_loss = train(model, device, optimizer, train_loader)
        train_writer.add_scalar('accuracy', train_acc, epoch)
        train_writer.add_scalar('loss', train_loss, epoch)
        test_acc, test_loss = test(model, device, test_loader)
        test_writer.add_scalar('accuracy', test_acc, epoch)
        test_writer.add_scalar('loss', test_loss, epoch)
        print('epoch: {:04d}, train acc: {:.4f}, test acc: {:.4f}'.format(epoch + 1, train_acc, test_acc))

    train_writer.close()
    test_writer.close()

    for c, data in enumerate(train_loader):
        if data.shape_id.cpu().item() in [0, 1, 2]:
            pred_class = predict(model, device, data)
            data.face = None
            export_colored_mesh(data, color=1,
                                output_path=run_path + '/train/' + str(data.shape_id.item()) + '_pred.off')


class Net(torch.nn.Module):
    def __init__(self, num_classes, pool):
        super(Net, self).__init__()
        self.pool = pool
        self.conv1 = GCNConv(5, 32)
        self.conv2 = GCNConv(32, 64)
        self.conv3 = GCNConv(64, 128)
        self.conv4 = GCNConv(128, 256)
        self.fc1 = torch.nn.Linear(256, num_classes)

    def forward(self, data):
        data.x = F.relu(self.conv1(data.x, data.edge_index))
        for _ in range(self.pool[0]):
            data = mesh_pool(data)
        data.x = F.relu(self.conv2(data.x, data.edge_index))
        for _ in range(self.pool[1]):
            data = mesh_pool(data)
        data.x = F.relu(self.conv3(data.x, data.edge_index))
        for _ in range(self.pool[2]):
            data = mesh_pool(data)
        data.x = F.relu(self.conv4(data.x, data.edge_index))
        for _ in range(self.pool[3]):
            data = mesh_pool(data)
        data.x = global_mean_pool(data.x, data.batch)
        data.x = F.log_softmax(self.fc1(data.x), dim=-1)
        return data.x


def train(model, device, optimizer, train_loader):
    model.train()
    correct = 0
    loss = 0

    for data in train_loader:
        optimizer.zero_grad()
        pred = model(data.to(device))
        pred_class = pred.max(dim=-1)[1]
        correct += pred_class.eq(data.y).sum().item() / len(data.y)
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
        correct += pred_class.eq(data.y).sum().item() / len(data.y)
        output = F.nll_loss(pred, data.y)
        loss += output.item()

    return correct / len(test_loader), loss / len(test_loader)


def predict(model, device, data):
    model.eval()
    pred = model(data.to(device))
    pred_class = pred.max(dim=-1)[1]
    print(data.shape_id.item(), 'gt:', data.y.cpu().item(), 'pred:', pred_class.item())
    return pred_class


def export_colored_mesh(data, labels=None, color=None, output_path=None):
    def get_incident_edges(edge, edges):
        return [x for x in edges if not set(x).isdisjoint(set(edge)) and not x == edge]

    def get_faces(edges):
        edges = [tuple(x) for x in edges]
        faces = set()
        for e in edges:
            i_edges = get_incident_edges(e, edges)
            # sort so that first point of edge is common point with e
            for c, ie in enumerate(i_edges):
                if ie[1] in e:
                    i_edges[c] = (ie[1], ie[0])

            for ie1 in i_edges:
                for ie2 in i_edges:
                    if ie1 != ie2 and ie1[1] == ie2[1]:
                        faces.add(tuple(sorted([e[0], e[1], ie1[1]])))
        return faces

        for c in cell_edges:
            cell = []
            for i in range(len(c)):
                if c[i][0] in c[(i + 1) % len(c)]:
                    cell.append(c[i][0])
                else:
                    cell.append(c[i][1])
            if len(set(cell)) == 4:
                faces.append(cell)

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
    if data.face is not None:
        faces = data.face.t().cpu().numpy()
    else:
        faces = get_faces(edges)
    if color is not None:
        # classification
        face_labels = np.full([len(faces)], color)
    else:
        # edge labels to face labels
        edge_labels = labels.cpu().numpy()
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
