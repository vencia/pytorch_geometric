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
    # only works with batchsize 1 for now else need to force unpool only between nodes of same mesh and change new batch
    if count == 0:
        return data

    print(data.shape_id.item())
    face_t = data.face.t()
    edge_index_t = data.edge_index.t()
    top_k = torch.topk(data.x[face_t].sum(dim=-1).sum(dim=-1), count)[1]
    unpool_faces_t = face_t[top_k]
    unpool_faces_with_edges_t = torch.stack(
        [torch.stack([x for x in edge_index_t if is_in(x, f).all() and x[0] < x[1]]) for f in unpool_faces_t])

    # update nodes (features, pos, batch)
    new_node_features = data.x[unpool_faces_with_edges_t].mean(dim=-2).mean(dim=-2)
    new_node_positions = data.pos[unpool_faces_with_edges_t].mean(dim=-2).mean(dim=-2)
    new_x = torch.cat([data.x, new_node_features])
    new_pos = torch.cat([data.pos, new_node_positions])
    new_nodes = torch.arange(start=data.num_nodes, end=data.num_nodes + count, device=torch.device('cuda'))
    new_batch = torch.cat(
        [data.batch, torch.full([count], data.batch[0], dtype=torch.long, device=torch.device('cuda'))])

    # update structure
    face_mask = torch.ones([len(face_t)], dtype=torch.uint8)
    face_mask[top_k] = 0
    non_unpool_faces = (face_t[face_mask]).t()

    new_edges_t = torch.cat([torch.stack(
        [torch.stack([new_nodes[c], x[0]]),
         torch.stack([new_nodes[c], x[1]]),
         torch.stack([new_nodes[c], x[2]])])
        for c, x in enumerate(unpool_faces_t)])

    new_faces_t = torch.cat([torch.stack(
        [torch.stack([new_nodes[c], x[0][0], x[0][1]]),
         torch.stack([new_nodes[c], x[1][0], x[1][1]]),
         torch.stack([new_nodes[c], x[2][0], x[2][1]])])
        for c, x in enumerate(unpool_faces_with_edges_t)])

    new_face = torch.cat([non_unpool_faces, new_faces_t.t()], dim=-1)
    new_edge_index = torch.cat([data.edge_index, new_edges_t.t(), other_direction(new_edges_t.t())], dim=-1)

    # update data
    data.x = new_x
    data.batch = new_batch
    data.pos = new_pos
    data.edge_index = new_edge_index
    data.face = new_face

    # mesh visualization attributes
    data.face_attr = torch.zeros([new_face.shape[1]], dtype=torch.uint8)
    data.face_attr[-len(new_faces_t):] = 1

    return data
