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
    assert len(edges) == 2
    return torch.stack([edges[1], edges[0]])


def get_border_faces_with_edges_t(num_borders, face_with_edges_t, face_borders_t):
    # border edges before non-border edges
    faces = [torch.cat([x[face_borders_t[c] == 1], x[face_borders_t[c] == 0]]) for c, x in
             enumerate(face_with_edges_t) if sum(face_borders_t[c]) == num_borders]
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
    face_t = data.face.t()
    edge_index_t = data.edge_index.t()
    face_with_edges_t = data.face_with_edges.t()

    top_k = torch.topk(data.x[face_t].sum(dim=-1).sum(dim=-1), count)[1]
    unpool_faces_with_edges_t = face_with_edges_t[top_k]
    unpool_edge_idxs = torch.unique(unpool_faces_with_edges_t.flatten())

    face_borders_t = torch.stack([is_in(x, unpool_edge_idxs) for x in face_with_edges_t])
    b1_faces_with_edges_t = get_border_faces_with_edges_t(1, face_with_edges_t, face_borders_t)
    b2_faces_with_edges_t = get_border_faces_with_edges_t(2, face_with_edges_t, face_borders_t)
    b3_faces_with_edges_t = get_border_faces_with_edges_t(3, face_with_edges_t, face_borders_t)
    assert len(b3_faces_with_edges_t) == len(unpool_faces_with_edges_t)

    edge_mask = torch.zeros([len(edge_index_t)], dtype=torch.uint8, device=torch.device('cuda'))
    edge_mask[unpool_edge_idxs] = 1
    unpool_edges_t = edge_index_t[edge_mask]  # one-directional
    other_dir_edge_mask = sum([(edge_index_t == x).all(-1) for x in other_direction(unpool_edges_t.t()).t()])
    non_unpool_edges_t = edge_index_t[edge_mask + other_dir_edge_mask == 0].view(-1, 2)  # bidirectional

    edge_to_middle_point_dict = {}
    middle_point_to_edge_dict = {}
    for c, e in enumerate(unpool_edge_idxs):
        mp = c + data.num_nodes
        edge_to_middle_point_dict[e.item()] = torch.tensor(mp, device=torch.device('cuda'))
        middle_point_to_edge_dict[mp] = e

    # update nodes (features, pos, batch)
    new_node_features = data.x[unpool_edges_t].mean(dim=-2)
    new_node_positions = data.pos[unpool_edges_t].mean(dim=-2)
    new_x = torch.cat([data.x, new_node_features])
    new_pos = torch.cat([data.pos, new_node_positions])
    new_node_idxs = torch.arange(start=data.num_nodes, end=data.num_nodes + len(unpool_edges_t),
                                 device=torch.device('cuda'))
    new_batch = torch.cat([data.batch, torch.full_like(new_node_idxs, data.batch[0])])  # only works for batchsize 1

    # update pooling faces structure
    new_to_new_faces = torch.stack([torch.stack(
        [edge_to_middle_point_dict[x[0].item()], edge_to_middle_point_dict[x[1].item()],
         edge_to_middle_point_dict[x[2].item()]]) for x in
        unpool_faces_with_edges_t]).t()

    old_to_new_edges1 = torch.stack([unpool_edges_t.t()[0], new_node_idxs])
    old_to_new_edges2 = torch.stack([unpool_edges_t.t()[1], new_node_idxs])
    new_to_new_edges = torch.cat([torch.stack([new_to_new_faces[0], new_to_new_faces[1]]),
                                  torch.stack([new_to_new_faces[0], new_to_new_faces[2]]),
                                  torch.stack([new_to_new_faces[1], new_to_new_faces[2]])], dim=-1)

    old_to_new_faces = torch.stack([torch.stack(
        [get_common_node(edge_index_t[middle_point_to_edge_dict[x[0].item()]],
                         edge_index_t[middle_point_to_edge_dict[x[1].item()]]), x[0], x[1]]) for
        x in new_to_new_edges.t()]).t()

    face_mask = torch.ones([len(face_t)], dtype=torch.uint8)
    face_mask[face_borders_t.sum(dim=-1) > 0] = 0
    remaining_faces = (face_t[face_mask]).t()
    new_edge_index = torch.cat(
        [non_unpool_edges_t.t(), old_to_new_edges1, other_direction(old_to_new_edges1), old_to_new_edges2,
         other_direction(old_to_new_edges2), new_to_new_edges, other_direction(new_to_new_edges)], dim=-1)
    new_face = torch.cat([remaining_faces, old_to_new_faces, new_to_new_faces], dim=-1)

    # update border faces structure
    if b1_faces_with_edges_t is not None:
        b1_edges = torch.stack(
            [torch.stack(
                [edge_to_middle_point_dict[x[0].item()],
                 get_common_node(edge_index_t[x[1]], edge_index_t[x[2]])])
                for c, x in enumerate(b1_faces_with_edges_t)]).t()
        b1_faces = torch.cat([torch.stack([torch.stack([edge_index_t[b1_faces_with_edges_t[c][0]][0], x[0], x[1]]),
                                           torch.stack([edge_index_t[b1_faces_with_edges_t[c][0]][1], x[0], x[1]])])
                              for c, x in enumerate(b1_edges.t())]).t()

        new_edge_index = torch.cat([new_edge_index, b1_edges, other_direction(b1_edges)], dim=-1)
        new_face = torch.cat([new_face, b1_faces], dim=-1)

    if b2_faces_with_edges_t is not None:
        b2_new_to_new_edges = torch.stack(
            [torch.stack([edge_to_middle_point_dict[x[0].item()], edge_to_middle_point_dict[x[1].item()]])
             for c, x in enumerate(b2_faces_with_edges_t)]).t()
        b2_old_to_new_edges = torch.stack(
            [torch.stack(
                [edge_to_middle_point_dict[x[0].item()],
                 get_common_node(edge_index_t[x[1]], edge_index_t[x[2]])])
                for c, x in enumerate(b2_faces_with_edges_t)]).t()
        b2_new_to_new_faces = torch.stack(
            [torch.stack(
                [get_common_node(edge_index_t[b2_faces_with_edges_t[c][0]], edge_index_t[b2_faces_with_edges_t[c][1]]),
                 x[0], x[1]]) for c, x in enumerate(b2_new_to_new_edges.t())]).t()
        b2_old_to_new_faces = torch.cat([torch.stack([
            torch.stack(
                [edge_index_t[x[0]][(edge_index_t[x[0]] != get_common_node(edge_index_t[x[0]], edge_index_t[x[1]]))][0],
                 b2_old_to_new_edges.t()[c][0], b2_old_to_new_edges.t()[c][1]]),
            torch.stack([b2_new_to_new_edges.t()[c][1], b2_old_to_new_edges.t()[c][0], b2_old_to_new_edges.t()[c][1]])])
            for c, x in enumerate(b2_faces_with_edges_t)]).t()

        new_edge_index = torch.cat([new_edge_index, b2_new_to_new_edges, other_direction(b2_new_to_new_edges),
                                    b2_old_to_new_edges, other_direction(b2_old_to_new_edges)], dim=-1)
        new_face = torch.cat([new_face, b2_new_to_new_faces, b2_old_to_new_faces], dim=-1)

    # update data
    data.x = new_x
    data.batch = new_batch
    data.pos = new_pos
    data.edge_index = new_edge_index
    data.face = new_face

    # mesh visualization attributes
    num_border_faces = (b1_faces.shape[1] if b1_faces_with_edges_t is not None else 0) + (
        b2_new_to_new_faces.shape[1] + b2_old_to_new_faces.shape[1] if b2_faces_with_edges_t is not None else 0)

    data.face_attr = torch.zeros([data.face.shape[1]], dtype=torch.uint8)
    data.face_attr[- old_to_new_faces.shape[1] - new_to_new_faces.shape[1] - num_border_faces: - num_border_faces] = 1
    data.face_attr[- num_border_faces:] = 2

    return data
