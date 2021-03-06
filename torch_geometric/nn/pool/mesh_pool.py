import torch


def is_in(ar1, ar2):
    return (ar1[..., None] == ar2).any(-1)


def mesh_pool(data, count, aggr='add'):
    """
    MeshCNN-like pooling but with node features instead of edge features.
    <https://arxiv.org/abs/1809.05910>
    """
    isolated_nodes_mask = is_in(torch.arange(data.num_nodes, device=torch.device('cuda')), data.edge_index[0]) != 1
    for _ in range(count):
        summed_features = data.x.sum(dim=-1) + isolated_nodes_mask.float() * 100000.0  # some high value
        x_min = summed_features.min(dim=0)[1]
        x_neighbors = data.edge_index[1][(data.edge_index[0] == x_min).nonzero()].squeeze(dim=1)
        y_min = x_neighbors[summed_features[x_neighbors].min(dim=0)[1]]
        y_neighbors = data.edge_index[1][(data.edge_index[0] == y_min).nonzero()].squeeze(dim=1)
        both_neighbors = [x for x in x_neighbors if x in y_neighbors]
        # if data.shape_id.cpu().item() in [1]:
        #     print('shape', data.shape_id.cpu().item(), 'edge', x_min.cpu().item(), y_min.cpu().item(), 'len both neighbors',
        #           len(both_neighbors))

        # contract edge (x_min, y_min)
        # new node gets idx of x_min with updated edges, y_min gets zeroed and gets deleted from edges
        data.pos[x_min] = (data.pos[x_min] + data.pos[y_min]) / 2
        if aggr == 'max':
            new_node_features = torch.max(data.x[x_min], data.x[y_min])  # TODO: doesn't work because of inplace operation?
        elif aggr == 'mean':
            new_node_features = (data.x[x_min] + data.x[y_min]) / 2
        else:
            assert aggr == 'add'
            new_node_features = data.x[x_min] + data.x[y_min]
        data.x[x_min] = new_node_features
        data.x[y_min] = torch.zeros_like(data.x[y_min])
        isolated_nodes_mask[y_min] = 1

        # remove now duplicate y_min edges
        row, col = data.edge_index
        mask = (((row == y_min) * (sum([(col == x) for x in both_neighbors]) + (col == x_min))) + (
                (col == y_min) * (sum([(row == x) for x in both_neighbors]) + (row == x_min)))) != 1
        mask = mask.unsqueeze(0).expand_as(data.edge_index)
        data.edge_index = data.edge_index[mask].view(2, -1)

        # update y_min edges to x_min
        data.edge_index[0][data.edge_index[0] == y_min] = x_min
        data.edge_index[1][data.edge_index[1] == y_min] = x_min

    return data
