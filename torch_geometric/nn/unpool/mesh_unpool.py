import torch


def is_in(ar1, ar2):
    return (ar1[..., None] == ar2).any(-1)


def edges_on_same_face(edges, faces):
    return any([all([is_in(e, f).all(-1) for e in edges]) for f in faces])


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


def mesh_unpool(data, count):
    # TODO: instead of count a threshold?
    # only works with batchsize 1 for now, else need to force unpool only between nodes of same mesh
    if count == 0:
        return data

    print(data.shape_id.item())
    top_k = torch.topk(data.x[data.face.t()].sum(dim=-1).sum(dim=-1), count)[1]
    unpool_faces_t = data.face.t()[top_k]
    unpool_faces_with_edge_idxs_t = torch.tensor([[x for x in range(len(data.edge_index.t())) if
                                                   is_in(data.edge_index.t()[x], f).all() and data.edge_index.t()[x][
                                                       0] <
                                                   data.edge_index.t()[x][1]] for f in unpool_faces_t],
                                                 device=torch.device('cuda'))

    edge_mask = torch.zeros([len(data.edge_index.t())], dtype=torch.uint8)
    edge_mask[torch.unique(unpool_faces_with_edge_idxs_t.flatten())] = 1
    unpool_edges_t = data.edge_index.t()[edge_mask]

    # update nodes (features, pos, batch)
    new_node_features = data.x[unpool_edges_t].mean(dim=-2)
    new_node_positions = data.pos[unpool_edges_t].mean(dim=-2)
    new_x = torch.cat([data.x, new_node_features])
    new_pos = torch.cat([data.pos, new_node_positions])
    new_node_idxs = torch.arange(start=data.num_nodes, end=data.num_nodes + len(unpool_edges_t),
                                 device=torch.torch.device('cuda'))
    new_batch = torch.cat([data.batch, torch.full_like(new_node_idxs, data.batch[0])])  # only works for batchsize 1

    # update edges
    non_unpool_edges = data.edge_index.t()[edge_mask == 0].t()
    old_to_new_edges1 = torch.stack([unpool_edges_t.t()[0], new_node_idxs])
    old_to_new_edges2 = torch.stack([unpool_edges_t.t()[1], new_node_idxs])
    new_to_new_edges = torch.tensor([(new_node_idxs[x], new_node_idxs[y]) for y in range(len(new_node_idxs))
                                     for x in range(y + 1, len(new_node_idxs)) if
                                     edges_on_same_face([unpool_edges_t[x], unpool_edges_t[y]], unpool_faces_t)],
                                    device=torch.device('cuda')).t()
    new_edge_index = torch.cat(
        [non_unpool_edges, old_to_new_edges1, other_direction(old_to_new_edges1), old_to_new_edges2,
         other_direction(old_to_new_edges2), new_to_new_edges, other_direction(new_to_new_edges)], dim=-1)

    # update faces

    # TODO: make way faster, don't need dict structure anyway
    face2edge_node = {}
    for i, f in enumerate(unpool_faces_with_edge_idxs_t):
        for j, e in enumerate(f):
            face2edge_node[(i, j)] = \
                (unpool_edges_t == data.edge_index.t()[unpool_faces_with_edge_idxs_t][i][j]).all(-1).nonzero()[
                    0] + data.num_nodes

    face_mask = torch.ones([len(data.face.t())], dtype=torch.uint8)
    face_mask[top_k] = 0
    non_unpool_faces = (data.face.t()[face_mask]).t()
    old_to_new_faces = torch.tensor(
        [(x, y, get_common_node(unpool_edges_t[x - data.num_nodes], unpool_edges_t[y - data.num_nodes])) for
         (x, y) in new_to_new_edges.t()],
        device=torch.device('cuda')).t()
    new_to_new_faces = torch.tensor(list((face2edge_node.values())), device=torch.device('cuda')).view(
        len(unpool_faces_t), -1).t()
    new_face = torch.cat([non_unpool_faces, old_to_new_faces, new_to_new_faces], dim=-1)

    data.x = new_x
    data.batch = new_batch
    data.pos = new_pos
    data.edge_index = new_edge_index
    data.face = new_face

    data.face_attr = torch.zeros([data.face.shape[1]], dtype=torch.uint8)
    data.face_attr[-old_to_new_faces.shape[1] - new_to_new_faces.shape[1]:] = 1

    return data
