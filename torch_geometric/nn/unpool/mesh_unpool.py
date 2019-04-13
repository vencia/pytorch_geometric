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


def mesh_unpool(data, count):
    # TODO: instead of count a threshold?
    # only works with batchsize 1 for now, else need to force unpool only between nodes of same mesh
    if count == 0:
        return data

    print(data.shape_id.item())
    top_k = torch.topk(data.x[data.face.t()].sum(dim=-1).sum(dim=-1), count)[1]
    unpool_faces = data.face.t()[top_k]
    unpool_nodes = torch.unique(unpool_faces.flatten())
    row, col = data.edge_index
    unpool_edges_mask = sum(
        [(col == unpool_nodes[x]) * (row == unpool_nodes[y]) for y in range(len(unpool_nodes)) for x in
         range(y + 1, len(unpool_nodes))]) > 0
    unpool_edges_mask = unpool_edges_mask.unsqueeze(0).expand_as(data.edge_index)
    unpool_edges = data.edge_index[unpool_edges_mask].view(2, -1)
    unpool_edges_t = unpool_edges.t()

    # update nodes (features, pos, batch)
    new_node_features = data.x[unpool_edges_t].mean(dim=-2)
    new_node_positions = data.pos[unpool_edges_t].mean(dim=-2)
    new_x = torch.cat([data.x, new_node_features])
    new_pos = torch.cat([data.pos, new_node_positions])
    new_node_idxs = torch.arange(start=data.num_nodes - 1, end=data.num_nodes + unpool_edges.shape[1] - 1,
                                 device=torch.torch.device('cuda'))
    new_batch = torch.cat([data.batch, torch.full_like(new_node_idxs, data.batch[0])])  # only works for batchsize 1

    # update edges
    non_unpool_edges = data.edge_index[unpool_edges_mask == 0].view(2, -1)
    old_to_new_edges = torch.stack([unpool_edges[0], new_node_idxs])
    # new_unpool_edges2 = torch.stack([new_node_idxs, unpool_edges[1]])
    new_to_new_edges = torch.tensor([(new_node_idxs[x], new_node_idxs[y]) for y in range(len(new_node_idxs))
                                     for x in range(y + 1, len(new_node_idxs)) if
                                     edges_on_same_face([unpool_edges_t[x], unpool_edges_t[y]], unpool_faces)],
                                    device=torch.device('cuda')).t()
    new_edge_index = torch.cat([non_unpool_edges, old_to_new_edges, new_to_new_edges], dim=-1)

    # update faces
    face_mask = torch.ones([len(data.face.t())], dtype=torch.uint8)
    face_mask[top_k] = 0
    non_unpool_faces = (data.face.t()[face_mask]).t()
    new_node_faces = torch.tensor(
        [(x, y, get_common_node(unpool_edges_t[x - data.num_nodes + 1], unpool_edges_t[y - data.num_nodes + 1])) for
         (x, y) in new_to_new_edges.t()],
        device=torch.device('cuda')).t()
    new_face = torch.cat([non_unpool_faces, new_node_faces], dim=-1)

    data.x = new_x
    data.batch = new_batch
    data.pos = new_pos
    data.edge_index = new_edge_index
    data.face = new_face

    return data
