import torch


def get_common_nodes(e1, e2):
    return e1[sum((e1 == e2, e1 == torch.stack((e2[..., 1], e2[..., 0]), dim=-1)))]


def other_direction(edges):
    assert edges.shape[1] == 2
    return torch.stack((edges[..., 1], edges[..., 0]), dim=-1)


def get_border_face_edges(num_borders, face_edges, face_borders):
    # border edges before non-border edges
    face_cases = face_borders.sum(dim=-1)
    case_mask = face_cases == num_borders
    relevant_faces = face_edges[case_mask]
    if len(relevant_faces) == 0:
        return None
    relevant_face_borders = face_borders[case_mask]
    sorted_faces = torch.cat(
        (relevant_faces.flatten()[(relevant_face_borders.flatten() == 1)].view(len(relevant_faces), -1),
         relevant_faces.flatten()[(relevant_face_borders.flatten() == 0)].view(len(relevant_faces), -1)), dim=-1)
    return sorted_faces


def mesh_unpool(data, count):
    # TODO: instead of count a threshold?
    # only works with batchsize 1 for now, else need to force unpool only between nodes of same mesh
    if count == 0:
        return data

    print(data.shape_id.item())
    faces = data.face.t()
    edge_index = data.edge_index.t()
    face_edges = data.face_edges.t()

    top_k = torch.topk(data.x[faces].sum(dim=-1).sum(dim=-1), count)[1]
    unpool_edge_idxs, unpool_edge_map = torch.unique(face_edges[top_k].flatten(), return_inverse=True)
    edge_to_mp = torch.zeros(len(edge_index), dtype=torch.long, device=torch.device('cuda'))
    edge_to_mp[face_edges[top_k].flatten()] = unpool_edge_map + data.num_nodes
    unpool_edges = edge_index[unpool_edge_idxs]

    face_borders = (face_edges.flatten()[..., None] == unpool_edge_idxs).any(-1).view(len(face_edges), -1)

    b1_face_edges = get_border_face_edges(1, face_edges, face_borders)
    b2_face_edges = get_border_face_edges(2, face_edges, face_borders)
    b3_face_edges = get_border_face_edges(3, face_edges, face_borders)

    # update nodes (features, pos, batch)
    new_node_features = data.x[unpool_edges].mean(dim=-2)
    new_node_positions = data.pos[unpool_edges].mean(dim=-2)
    new_x = torch.cat((data.x, new_node_features))
    new_pos = torch.cat((data.pos, new_node_positions))
    new_node_idxs = torch.arange(start=data.num_nodes, end=data.num_nodes + len(unpool_edges),
                                 device=torch.device('cuda'))
    new_batch = torch.cat((data.batch, torch.full_like(new_node_idxs, data.batch[0])))  # only works for batchsize 1

    # update pooling faces structure
    inner_faces = edge_to_mp[b3_face_edges.flatten()].view(-1, 3)

    outer_faces = torch.cat((
        torch.stack(
            (get_common_nodes(edge_index[b3_face_edges[..., 0]], edge_index[b3_face_edges[..., 1]]),
             inner_faces[..., 0], inner_faces[..., 1])),
        torch.stack(
            (get_common_nodes(edge_index[b3_face_edges[..., 1]], edge_index[b3_face_edges[..., 2]]),
             inner_faces[..., 1], inner_faces[..., 2])),
        torch.stack(
            (get_common_nodes(edge_index[b3_face_edges[..., 0]], edge_index[b3_face_edges[..., 2]]),
             inner_faces[..., 0], inner_faces[..., 2]))), dim=-1).t()

    face_mask = torch.ones(len(faces), dtype=torch.uint8)
    face_mask[face_borders.sum(dim=-1) > 0] = 0
    remaining_faces = (faces[face_mask])
    new_faces = torch.cat((remaining_faces, outer_faces, inner_faces))

    # update border faces structure
    if b1_face_edges is not None:
        b1_mps_0 = edge_to_mp[b1_face_edges[..., 0]]
        b1_opp_0 = get_common_nodes(edge_index[b1_face_edges[..., 1]], edge_index[b1_face_edges[..., 2]])

        b1_faces = torch.cat((
            torch.stack((b1_mps_0, edge_index[b1_face_edges[..., 0]][..., 0], b1_opp_0)),
            torch.stack((b1_mps_0, edge_index[b1_face_edges[..., 0]][..., 1], b1_opp_0))), dim=-1).t()

        new_faces = torch.cat((new_faces, b1_faces))

        if b2_face_edges is not None:
            b2_mps_0 = edge_to_mp[b2_face_edges[..., 0]]
            b2_mps_1 = edge_to_mp[b2_face_edges[..., 1]]
            b2_opp_0 = get_common_nodes(edge_index[b2_face_edges[..., 1]], edge_index[b2_face_edges[..., 2]])
            b2_opp_1 = get_common_nodes(edge_index[b2_face_edges[..., 0]], edge_index[b2_face_edges[..., 2]])
            b2_opp_2 = get_common_nodes(edge_index[b2_face_edges[..., 0]], edge_index[b2_face_edges[..., 1]])

            b2_mp_to_mp_faces = torch.stack((b2_mps_0, b2_mps_1, b2_opp_2)).t()
            b2_middle_faces = torch.stack((b2_mps_0, b2_mps_1, b2_opp_0)).t()
            b2_other_faces = torch.stack((b2_mps_0, b2_opp_0, b2_opp_1)).t()

            new_faces = torch.cat((new_faces, b2_mp_to_mp_faces, b2_middle_faces, b2_other_faces))

        p0s = new_faces[..., 0]
        p1s = new_faces[..., 1]
        p2s = new_faces[..., 2]

        new_faces_with_edges = torch.transpose(torch.transpose(torch.stack(
            (torch.stack((p0s, p1s)), torch.stack((p1s, p2s)), torch.stack((p0s, p2s)),
             torch.stack((p1s, p0s)), torch.stack((p2s, p1s)), torch.stack((p2s, p0s)))), 0, 1), 0, 2)

        new_edges, face_to_edge_map = torch.unique(new_faces_with_edges.reshape(-1, 2), return_inverse=True, dim=0)
        new_face_edges = face_to_edge_map.view(-1, 6)[..., :3]  # one-directional

        # update data
        data.x = new_x
        data.batch = new_batch
        data.pos = new_pos
        data.edge_index = new_edges.t()  # new_edge_index.t()
        data.face = new_faces.t()
        data.face_edges = new_face_edges.t()

        # mesh visualization attributes
        num_border_faces = (len(b1_faces) if b1_face_edges is not None else 0) + (
            len(b2_mp_to_mp_faces) + len(b2_middle_faces) + len(b2_other_faces) if b2_face_edges is not None else 0)

        data.face_attr = torch.zeros(len(new_faces), dtype=torch.uint8)
        data.face_attr[- len(outer_faces) - len(inner_faces) - num_border_faces: - num_border_faces] = 1
        data.face_attr[- num_border_faces:] = 2

    return data
