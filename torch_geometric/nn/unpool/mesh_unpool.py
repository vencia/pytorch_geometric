import torch


def is_in(ar1, ar2):
    return (ar1[..., None] == ar2).any(-1)


def get_common_node(e1, e2):
    if e1[0] in e2:
        return e1[0]
    elif e1[1] in e2:
        return e1[1]
    else:
        return None


def other_direction(edges):
    assert edges.shape[1] == 2
    return torch.stack([edges[..., 1], edges[..., 0]], dim=-1)


def get_border_face_edges(num_borders, face_edges_t, face_borders_t):
    # border edges before non-border edges
    faces = [torch.cat([x[face_borders_t[c] == 1], x[face_borders_t[c] == 0]]) for c, x in
             enumerate(face_edges_t) if sum(face_borders_t[c]) == num_borders]
    if len(faces) > 0:
        return torch.stack(faces)
    else:
        return None


def mesh_unpool(data, count):
    # TODO: instead of count a threshold?
    # only works with batchsize 1 for now, else need to force unpool only between nodes of same mesh
    if count == 0:
        return data

    print(data.shape_id.item())
    faces = data.face.t()
    edge_index = data.edge_index.t()
    face_edges = data.face_with_edges.t()

    top_k = torch.topk(data.x[faces].sum(dim=-1).sum(dim=-1), count)[1]
    unpool_face_edges = face_edges[top_k]
    unpool_edge_idxs = torch.unique(unpool_face_edges.flatten())

    face_borders = torch.stack([is_in(x, unpool_edge_idxs) for x in face_edges])
    b1_face_edges = get_border_face_edges(1, face_edges, face_borders)
    b2_face_edges = get_border_face_edges(2, face_edges, face_borders)
    b3_face_edges = get_border_face_edges(3, face_edges, face_borders)
    assert len(b3_face_edges) == len(unpool_face_edges)

    edge_mask = torch.zeros([len(edge_index)], dtype=torch.uint8, device=torch.device('cuda'))
    edge_mask[unpool_edge_idxs] = 1
    unpool_edges = edge_index[edge_mask]  # one-directional
    other_dir_edge_mask = sum([(edge_index == x).all(-1) for x in other_direction(unpool_edges)])
    non_unpool_edges = edge_index[edge_mask + other_dir_edge_mask == 0].view(-1, 2)  # bidirectional

    edge_to_middle_point_dict = {}
    middle_point_to_edge_dict = {}
    for c, e in enumerate(unpool_edge_idxs):
        mp = c + data.num_nodes
        edge_to_middle_point_dict[e.item()] = torch.tensor(mp, device=torch.device('cuda'))
        middle_point_to_edge_dict[mp] = e

    # update nodes (features, pos, batch)
    new_node_features = data.x[unpool_edges].mean(dim=-2)
    new_node_positions = data.pos[unpool_edges].mean(dim=-2)
    new_x = torch.cat([data.x, new_node_features])
    new_pos = torch.cat([data.pos, new_node_positions])
    new_node_idxs = torch.arange(start=data.num_nodes, end=data.num_nodes + len(unpool_edges),
                                 device=torch.device('cuda'))
    new_batch = torch.cat([data.batch, torch.full_like(new_node_idxs, data.batch[0])])  # only works for batchsize 1

    # update pooling faces structure
    new_to_new_faces = torch.stack([torch.stack(
        [edge_to_middle_point_dict[x[0].item()], edge_to_middle_point_dict[x[1].item()],
         edge_to_middle_point_dict[x[2].item()]]) for x in
        unpool_face_edges])

    old_to_new_edges = torch.cat(
        [torch.stack([unpool_edges.t()[0], new_node_idxs]).t(), torch.stack([unpool_edges.t()[1], new_node_idxs]).t()])
    new_to_new_edges = torch.cat([torch.stack([new_to_new_faces.t()[0], new_to_new_faces.t()[1]]),
                                  torch.stack([new_to_new_faces.t()[0], new_to_new_faces.t()[2]]),
                                  torch.stack([new_to_new_faces.t()[1], new_to_new_faces.t()[2]])], dim=-1).t()

    old_to_new_faces = torch.stack([torch.stack(
        [get_common_node(edge_index[middle_point_to_edge_dict[x[0].item()]],
                         edge_index[middle_point_to_edge_dict[x[1].item()]]), x[0], x[1]]) for
        x in new_to_new_edges])

    face_mask = torch.ones([len(faces)], dtype=torch.uint8)
    face_mask[face_borders.sum(dim=-1) > 0] = 0
    remaining_faces = (faces[face_mask])
    new_edge_index = torch.cat(
        [non_unpool_edges, old_to_new_edges, other_direction(old_to_new_edges), new_to_new_edges,
         other_direction(new_to_new_edges)])
    new_faces = torch.cat([remaining_faces, old_to_new_faces, new_to_new_faces])

    # update border faces structure
    if b1_face_edges is not None:
        b1_edges = torch.stack(
            [torch.stack(
                [edge_to_middle_point_dict[x[0].item()],
                 get_common_node(edge_index[x[1]], edge_index[x[2]])])
                for c, x in enumerate(b1_face_edges)])
        b1_faces = torch.cat([torch.stack([torch.stack([edge_index[b1_face_edges[c][0]][0], x[0], x[1]]),
                                           torch.stack([edge_index[b1_face_edges[c][0]][1], x[0], x[1]])])
                              for c, x in enumerate(b1_edges)])

        new_edge_index = torch.cat([new_edge_index, b1_edges, other_direction(b1_edges)])
        new_faces = torch.cat([new_faces, b1_faces])

    if b2_face_edges is not None:
        b2_new_to_new_edges = torch.stack(
            [torch.stack([edge_to_middle_point_dict[x[0].item()], edge_to_middle_point_dict[x[1].item()]])
             for c, x in enumerate(b2_face_edges)])
        b2_old_to_new_edges = torch.stack(
            [torch.stack(
                [edge_to_middle_point_dict[x[0].item()],
                 get_common_node(edge_index[x[1]], edge_index[x[2]])])
                for c, x in enumerate(b2_face_edges)])
        b2_new_to_new_faces = torch.stack(
            [torch.stack(
                [get_common_node(edge_index[b2_face_edges[c][0]], edge_index[b2_face_edges[c][1]]),
                 x[0], x[1]]) for c, x in enumerate(b2_new_to_new_edges)])
        b2_old_to_new_faces = torch.cat([torch.stack([
            torch.stack(
                [edge_index[x[0]][(edge_index[x[0]] != get_common_node(edge_index[x[0]], edge_index[x[1]]))][0],
                 b2_old_to_new_edges[c][0], b2_old_to_new_edges[c][1]]),
            torch.stack([b2_new_to_new_edges[c][1], b2_old_to_new_edges[c][0], b2_old_to_new_edges[c][1]])])
            for c, x in enumerate(b2_face_edges)])

        new_edge_index = torch.cat([new_edge_index, b2_new_to_new_edges, other_direction(b2_new_to_new_edges),
                                    b2_old_to_new_edges, other_direction(b2_old_to_new_edges)])
        new_faces = torch.cat([new_faces, b2_new_to_new_faces, b2_old_to_new_faces])

    # update data
    data.x = new_x
    data.batch = new_batch
    data.pos = new_pos
    data.edge_index = new_edge_index.t()
    data.face = new_faces.t()

    # mesh visualization attributes
    num_border_faces = (len(b1_faces) if b1_face_edges is not None else 0) + (
        len(b2_new_to_new_faces) + len(b2_old_to_new_faces) if b2_face_edges is not None else 0)

    data.face_attr = torch.zeros([len(new_faces)], dtype=torch.uint8)
    data.face_attr[- len(old_to_new_faces) - len(new_to_new_faces) - num_border_faces: - num_border_faces] = 1
    data.face_attr[- num_border_faces:] = 2

    return data
