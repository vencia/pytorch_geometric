import os
import os.path as osp
import shutil
import numpy as np
import click
from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
from torch_geometric.datasets.meshes import Meshes
import torch_geometric.transforms as T
from torch_geometric.transforms.face_to_edge import FaceToEdge
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn.pool.mesh_pool import mesh_pool
from torch_geometric.nn.unpool.mesh_unpool import mesh_unpool
from torch_geometric.utils.chamfer_distance import ChamferDistance
from torch_geometric.utils.undirected import to_undirected
import matplotlib as mpl
import networkx as nx
import matplotlib.pyplot as plt

mpl.use('Agg')


class AddMeshStructureTransform(object):
    def __call__(self, data):
        def is_in(ar1, ar2):
            return (ar1[..., None] == ar2).any(-1)

        print('add mesh structure for shape', data.shape_id.numpy())
        face_t = data.face.t()
        edge_index_t = data.edge_index.t()
        face_edges_t = torch.tensor([[x for x in range(len(edge_index_t)) if
                                      is_in(edge_index_t[x], f).all() and edge_index_t[x][0] <
                                      edge_index_t[x][1]] for f in face_t],
                                    device=torch.device('cuda'))
        data.face_edges = face_edges_t.t()

        gt_edge_index = torch.cat([data.gt_face[:2], data.gt_face[1:], data.gt_face[::2]], dim=1)
        gt_edge_index = to_undirected(gt_edge_index, num_nodes=data.num_nodes)
        data.gt_edge_index = gt_edge_index
        return data


class Rescale(object):
    r"""Centers and normalizes ip and gt node positions to the interval :math:`(-1, 1)`.
    """

    def __call__(self, data):
        def normalize(x):
            x *= 2
            x -= 1
            # x -= x.mean(dim=0, keepdim=True)
            # scale = (1 / x.abs().max()) * 0.999999
            # x *= scale
            return x

        data.x = normalize(data.x)
        data.gt_x = normalize(data.gt_x)
        return data


@click.command()
@click.option('--epochs', default=1)
@click.option('--lr', default=0.00000000002)
@click.option('--pool', nargs=4, default=(0, 0, 0, 0))
@click.option('--heatmap', is_flag=True)
@click.option('--vertices', default=6)
@click.option('--hole', is_flag=True)
def main(epochs, lr, pool, heatmap, vertices, hole):
    print('pool', pool)
    data_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'meshes',
                         'v{}{}'.format(vertices, 'hole' if hole else ''))
    arguments = 'v{}{}_e{}_lr{}_p{}'.format(vertices, 'hole' if hole else '', epochs, lr,
                                            '-'.join([str(x) for x in pool]))
    run_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'runs', arguments)
    if os.path.exists(run_path):
        shutil.rmtree(run_path)
    os.makedirs(run_path)

    # pre_filter = lambda x: x.shape_id.item() < 10
    pre_transform = T.Compose(
        [Rescale(), FaceToEdge(remove_faces=False), AddMeshStructureTransform()])
    train_dataset = Meshes(data_path, vertices=vertices, hole=hole, train=True, pre_transform=pre_transform)
    test_dataset = Meshes(data_path, vertices=vertices, hole=hole, train=False, pre_transform=pre_transform)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = Net(pool, run_path, heatmap).to(device)

    train_writer = SummaryWriter(run_path + '/train')
    test_writer = SummaryWriter(run_path + '/test')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    visualize_gt(10, train_loader, test_loader, run_path, heatmap)

    for epoch in range(epochs):
        train_loss = train(model, device, optimizer, train_loader, train_writer, epoch)
        test_loss = train(model, device, optimizer, test_loader, test_writer, epoch, test=True)
        print('epoch: {:04d}, train loss: {:.4f}, test loss: {:.4f}'.format(epoch + 1, train_loss, test_loss))

    train_writer.close()
    test_writer.close()

    visualize_pred(10, train_loader, test_loader, run_path, model, device)


class Net(torch.nn.Module):
    def __init__(self, pool, run_path, heatmap):
        super(Net, self).__init__()
        self.pool = pool
        self.run_path = run_path
        self.heatmap = heatmap

        self.conv1 = GCNConv(3, 32)
        self.conv2 = GCNConv(32, 64)
        self.conv3 = GCNConv(64, 32)
        self.conv4 = GCNConv(32, 3)

    def forward(self, data):
        data.x = F.relu(self.conv1(data.x, data.edge_index))
        data = mesh_unpool(data, self.pool[0])
        data.x = F.relu(self.conv2(data.x, data.edge_index))
        data = mesh_unpool(data, self.pool[1])
        data.x = F.relu(self.conv3(data.x, data.edge_index))
        data = mesh_unpool(data, self.pool[2])
        data.x = F.relu(self.conv4(data.x, data.edge_index))
        data = mesh_unpool(data, self.pool[3])
        if self.heatmap:
            data.face_attr = data.x[data.face.t()].sum(dim=-1).sum(dim=-1)
        # data.x = global_mean_pool(data.x, data.batch)
        # data.x = F.relu(self.fc1(data.x))
        # data.x = F.log_softmax(self.fc2(data.x), dim=-1)
        return data.x


def train(model, device, optimizer, data_loader, summary_writer, epoch, test=False):
    if test:
        model.eval()
    else:
        model.train()
    loss = 0

    for c, data in enumerate(data_loader):
        optimizer.zero_grad()
        pred = model(data.to(device))
        output = nn_loss(pred, data.gt_x)
        loss += output.item()
        if not test:
            output.backward()
            optimizer.step()

        step = len(data_loader) * epoch + c
        if epoch % 10 == 0 and data.shape_id.cpu().numpy()[0] == 2:
            graph_fig = get_graph_figure(pred, data.edge_index.t(), data.gt_x, data.gt_edge_index.t(), data.shape_id)
            summary_writer.add_figure('mesh', graph_fig, global_step=step)

    loss /= len(data_loader)
    summary_writer.add_scalar('loss', loss, epoch)
    # summary_writer.add_scalar('accuracy', acc, epoch)
    return loss


def predict(model, device, data):
    model.eval()
    pred = model(data.to(device))
    pred_class = pred.max(dim=-1)[1]
    # print(data.shape_id.item(), 'gt:', data.y.cpu().item(), 'pred:', pred_class.item())
    return pred_class


def nn_loss(x, y):
    chamfer_dist = ChamferDistance()
    dist1, dist2 = chamfer_dist(x.unsqueeze(dim=0), y.unsqueeze(dim=0))
    return torch.mean(dist2)  # nn-distance for all gt vertices


def visualize_gt(count, train_loader, test_loader, run_path, heatmap):
    for data in train_loader:
        if data.shape_id.cpu().numpy()[0] < count:
            if heatmap:
                data.face_attr = data.x[data.face.t()].sum(dim=-1).sum(dim=-1)
            export_colored_mesh(data.gt_x, data.gt_face, color=0,
                                output_path=run_path + '/train/{}_{}_gt.off'.format(data.shape_id.cpu().numpy()[0],
                                                                                    data.shape_id.cpu().numpy()[1]))
            export_colored_mesh(data.x, data.face, color=0,
                                output_path=run_path + '/train/{}_{}_ip.off'.format(data.shape_id.cpu().numpy()[0],
                                                                                    data.shape_id.cpu().numpy()[1]))
    for data in test_loader:
        if data.shape_id.cpu().numpy()[0] < count:
            if heatmap:
                data.face_attr = data.x[data.face.t()].sum(dim=-1).sum(dim=-1)
            export_colored_mesh(data.gt_x, data.gt_face, color=0,
                                output_path=run_path + '/test/{}_{}_gt.off'.format(data.shape_id.cpu().numpy()[0],
                                                                                   data.shape_id.cpu().numpy()[1]))
            export_colored_mesh(data.x, data.face, color=0,
                                output_path=run_path + '/train/{}_{}_ip.off'.format(data.shape_id.cpu().numpy()[0],
                                                                                    data.shape_id.cpu().numpy()[1]))


def visualize_pred(count, train_loader, test_loader, run_path, model, device):
    for data in train_loader:
        if data.shape_id.cpu().numpy()[0] < count:
            predict(model, device, data)
            export_colored_mesh(data.x, data.face, color=1,
                                output_path=run_path + '/train/{}_{}_pred.off'.format(data.shape_id.cpu().numpy()[0],
                                                                                      data.shape_id.cpu().numpy()[1]))
    for data in test_loader:
        if data.shape_id.cpu().numpy()[0] < count:
            predict(model, device, data)
            export_colored_mesh(data.x, data.face, color=1,
                                output_path=run_path + '/test/{}_{}_pred.off'.format(data.shape_id.cpu().numpy()[0],
                                                                                     data.shape_id.cpu().numpy()[1]))


def export_colored_mesh(vertices_tensor, faces_tensor, face_attr_tensor=None, color=None, output_path=None):
    def label_color(label):
        if label == 0:
            return ['0.0', '0.0', '1.0']
        elif label == 1:
            return ['0.0', '1.0', '0.0']
        elif label == 2:
            return ['1.0', '0.0', '0.0']
        else:
            assert label == 3
            return ['1.0', '1.0', '0.0']

    def heatmap_color(x, minimum, maximum):
        def format(x):
            return '%.3f' % round(x, 3)

        ratio = 2 * (x - minimum) / (maximum - minimum)
        b = max(0, (1 - ratio))
        r = max(0, (ratio - 1))
        g = 1.0 - b - r
        return [format(r), format(g), format(b)]

    vertices = vertices_tensor.detach().cpu().numpy()
    faces = faces_tensor.t().cpu().numpy()

    if face_attr_tensor:
        face_attr = face_attr_tensor.detach().cpu().numpy()
        if face_attr.dtype == np.float32:
            face_labels = [heatmap_color(x, face_attr.min(), face_attr.max()) for x in face_attr]
        else:
            face_labels = [label_color(x) for x in face_attr]
    else:
        labels = np.full([len(faces)], color)
        face_labels = [label_color(x) for x in labels]

    # write off file
    lines = ['OFF']
    lines.append('{} {} {}'.format(len(vertices), len(faces), 0))
    lines += [' '.join(str(y) for y in x) for x in vertices]
    lines += [' '.join(['3'] + [str(y) for y in x] + l) for x, l in zip(faces, face_labels)]
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def get_graph_figure(vertices, edges, gt_vertices, gt_edges, shape_id):
    def build_graph(v, e, color='b'):
        g = nx.Graph()
        for c, vertex in enumerate(v):
            g.add_node(c, pos=vertex[:2])
        for edge in e:
            g.add_edge(edge[0], edge[1], color=color)
        return g

    vertices = vertices.detach().cpu().numpy()
    edges = edges.cpu().numpy()
    gt_vertices = gt_vertices.cpu().numpy()
    gt_edges = gt_edges.cpu().numpy()
    shape_id = shape_id.cpu().numpy()
    mesh_name = '{}_{}'.format(shape_id[0], shape_id[1])

    graph = build_graph(vertices, edges)
    gt_graph = build_graph(gt_vertices, gt_edges, color='g')
    graph = nx.compose(graph, gt_graph)

    plt.title(mesh_name)
    pos = nx.get_node_attributes(graph, 'pos')
    colors = [graph[u][v]['color'] for u, v in graph.edges()]
    nx.draw_networkx(graph, pos, with_labels=False, node_size=8, width=1.5, edge_color=colors)
    return plt.figure(1)


if __name__ == '__main__':
    main()
