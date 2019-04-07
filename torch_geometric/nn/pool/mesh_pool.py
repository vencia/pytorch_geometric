def mesh_pool(data):
    summed_attributes = data.x.sum(dim=-1)
    x_min = summed_attributes.min(dim=0)[1]
    x_neighbors = data.edge_index[1][(data.edge_index[0] == x_min).nonzero()].squeeze(dim=1)
    if len(x_neighbors) == 0:  # isolated vertex
        new_data_x = data.x.clone()  # TODO: make faster
        new_data_x[x_min] = data.x.max()  # some high value # TODO: change, because then wrong global pooling etc.
        data.x = new_data_x
        return data
    # print('shape', data.shape_id.cpu().item(), 'edge', x_min.cpu().item(), x_neighbors.cpu().numpy())
    y_min = x_neighbors[summed_attributes[x_neighbors].min(dim=0)[1]]
    y_neighbors = data.edge_index[1][(data.edge_index[0] == y_min).nonzero()].squeeze(dim=1)
    both_neighbors = [x for x in x_neighbors if x in y_neighbors]
    # if data.shape_id.cpu().item() in [1]:
    #     print('shape', data.shape_id.cpu().item(), 'edge', x_min.cpu().item(), y_min.cpu().item(), 'len both neighbors',
    #           len(both_neighbors))

    # contract edge (x_min, y_min)
    # new node gets idx of x_min with updated edges, y_min gets dummy value and gets deleted from edges
    data.pos[x_min] = (data.pos[x_min] + data.pos[y_min]) / 2
    new_node_attr = (data.x[x_min] + data.x[y_min]) / 2
    new_data_x = data.x.clone()  # TODO: make faster
    new_data_x[x_min] = new_node_attr
    new_data_x[y_min] = data.x.max()  # some high value # TODO: change, because then wrong global pooling etc.
    data.x = new_data_x

    # remove now duplicate y_min edges
    row, col = data.edge_index
    mask = (((row == y_min) * (sum([(col == x) for x in both_neighbors]) + (col == x_min))) + (
            (col == y_min) * (sum([(row == x) for x in both_neighbors]) + (row == x_min)))) != 1
    mask = mask.unsqueeze(0).expand_as(data.edge_index)
    data.edge_index = data.edge_index[mask].view(2, -1)

    # update y_min edges to y_min
    data.edge_index[0][data.edge_index[0] == y_min] = x_min
    data.edge_index[1][data.edge_index[1] == y_min] = x_min

    return data
