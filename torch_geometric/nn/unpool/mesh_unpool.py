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


def mesh_unpool(data, count):
    # TODO: instead of count a threshold?
    # only works with batchsize 1 for now, else need to force unpool only between nodes of same mesh
    if count == 0:
        return data

    print(data.shape_id.item())
    face_t = data.face.t()
    edge_index_t = data.edge_index.t()
    top_k = torch.topk(data.x[face_t].sum(dim=-1).sum(dim=-1), count)[1]
    unpool_faces_t = face_t[top_k]
    unpool_faces_with_edge_idxs_t = torch.tensor([[x for x in range(len(edge_index_t)) if
                                                   is_in(edge_index_t[x], f).all() and edge_index_t[x][0] <
                                                   edge_index_t[x][1]] for f in unpool_faces_t],
                                                 device=torch.device('cuda'))

    unique_edges, unique_edges_map = torch.unique(unpool_faces_with_edge_idxs_t.flatten(), return_inverse=True)
    edge_mask = torch.zeros([len(edge_index_t)], dtype=torch.uint8, device=torch.device('cuda'))
    edge_mask[unique_edges] = 1
    unpool_edges_t = edge_index_t[edge_mask]  # one-directional
    other_dir_edge_mask = sum([(edge_index_t == x).all(-1) for x in other_direction(unpool_edges_t.t()).t()])
    non_unpool_edges_t = edge_index_t[edge_mask + other_dir_edge_mask == 0].view(-1, 2)  # bidirectional

    # update nodes (features, pos, batch)
    new_node_features = data.x[unpool_edges_t].mean(dim=-2)
    new_node_positions = data.pos[unpool_edges_t].mean(dim=-2)
    new_x = torch.cat([data.x, new_node_features])
    new_pos = torch.cat([data.pos, new_node_positions])
    new_node_idxs = torch.arange(start=data.num_nodes, end=data.num_nodes + len(unpool_edges_t),
                                 device=torch.device('cuda'))
    new_batch = torch.cat([data.batch, torch.full_like(new_node_idxs, data.batch[0])])  # only works for batchsize 1

    # update structure
    face_mask = torch.ones([len(face_t)], dtype=torch.uint8)
    face_mask[top_k] = 0
    non_unpool_faces = (face_t[face_mask]).t()
    new_to_new_faces = unique_edges_map.view(len(unpool_faces_t), -1).t() + data.num_nodes

    old_to_new_edges1 = torch.stack([unpool_edges_t.t()[0], new_node_idxs])
    old_to_new_edges2 = torch.stack([unpool_edges_t.t()[1], new_node_idxs])
    new_to_new_edges = torch.cat([torch.stack([new_to_new_faces[0], new_to_new_faces[1]]),
                                  torch.stack([new_to_new_faces[0], new_to_new_faces[2]]),
                                  torch.stack([new_to_new_faces[1], new_to_new_faces[2]])], dim=-1)
    new_edge_index = torch.cat(
        [non_unpool_edges_t.t(), old_to_new_edges1, other_direction(old_to_new_edges1), old_to_new_edges2,
         other_direction(old_to_new_edges2), new_to_new_edges, other_direction(new_to_new_edges)], dim=-1)

    old_to_new_faces = torch.tensor(
        [(get_common_node(unpool_edges_t[x[0] - data.num_nodes], unpool_edges_t[x[1] - data.num_nodes]), x[0], x[1]) for
         x in new_to_new_edges.t()],
        device=torch.device('cuda')).t()
    new_face = torch.cat([non_unpool_faces, old_to_new_faces, new_to_new_faces], dim=-1)

    # update data
    data.x = new_x
    data.batch = new_batch
    data.pos = new_pos
    data.edge_index = new_edge_index
    data.face = new_face

    # mesh visualization attributes
    data.face_attr = torch.zeros([data.face.shape[1]], dtype=torch.uint8)
    data.face_attr[-old_to_new_faces.shape[1] - new_to_new_faces.shape[1]:] = 1

    return data
