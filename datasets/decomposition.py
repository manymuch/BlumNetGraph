import cv2
import numpy as np
import random
import torch
from skimage import morphology
from collections import defaultdict
from torch.functional import F

EIGHT_NBRS = [(-1, -1), (0, -1), (1, -1),
              (-1, 0),           (1, 0),
              (-1, 1),  (0, 1),  (1, 1)]
VH4_NBRS = [      (0, -1),
            (-1, 0),     (1, 0),
                  (0,  1),      ]
X4_NBRS = [(-1, -1),  (1, -1),
           (-1,  1),  (1,  1)]
DIL_KERNER = np.zeros((3, 3), dtype=np.uint8)
DIL_KERNER[:, 1] += 1
DIL_KERNER[1, :] += 1
CONNECT_PT = 0
JUNCTION_PT = 1
END_PT = 2


def on_skeleton(nx, ny, h, w, sk_mask):
    out_side = min(nx, ny) < 0 or nx >= w or ny >= h
    if out_side:
        return False
    return sk_mask[ny, nx] != 0


def get_skeleton_keypts(skeleton):
    h, w = skeleton.shape[:2]
    yy, xx = np.where(skeleton > 0)
    points = []
    type2pts = defaultdict(set)
    pts2type = dict()
    for x, y in zip(xx.tolist(), yy.tolist()):
        vh4_pts = [
            (x + dx, y + dy) for dx, dy in VH4_NBRS
            if on_skeleton(x + dx, y + dy, h, w, skeleton)
        ]
        x4_pts = [
            (x + dx, y + dy) for dx, dy in X4_NBRS
            if on_skeleton(x + dx, y + dy, h, w, skeleton)
        ]
        type2pts[(len(vh4_pts), len(x4_pts))].add((x, y))
        points.append((x, y))
        pts2type[(x, y)] = (len(vh4_pts), len(x4_pts))
    return points, type2pts, pts2type


def walk_along_edge(start_pt, pts_set, pts2type, skeleton_cp, h, w):
    (x, y) = start_pt
    is_mid_pt = (sum(pts2type[(x, y)]) == 2)
    skeleton_cp[y, x] = 0
    pts_set.remove((x, y))

    # set the start point of edges for given crossing point
    edges_cur = []
    for dx, dy in EIGHT_NBRS:
        if not on_skeleton(x + dx, y + dy, h, w, skeleton_cp):
            continue
        edges_cur.append(
            [(x, y), (x + dx, y + dy)]  # add an edge
        )
        skeleton_cp[y + dy, x + dx] = 0
        pts_set.remove((x + dx, y + dy))

    # walk along each edge at current start point
    for edge in edges_cur:
        (x, y) = edge[-1]
        # walk along current edge
        while True:
            next_nb_pts = [
                (x + dx, y + dy) for dx, dy in EIGHT_NBRS
                if on_skeleton(x + dx, y + dy, h, w, skeleton_cp)
            ]
            if len(next_nb_pts) == 1:
                x, y = next_nb_pts[0]
                nb_pt = (x, y)
                edge.append(nb_pt)
                skeleton_cp[y, x] = 0
                pts_set.remove((x, y))
            else:
                # walk to the edge end while meeting the next crossing
                break
    if is_mid_pt:
        assert len(edges_cur) <= 2
        if len(edges_cur) == 2:
            combine_edge = edges_cur[1][::-1] + edges_cur[0][1:]
            edges_cur = [combine_edge]
    return edges_cur


def split_skeleton(skeleton, points=None, type2pts=None, pts2type=None):
    edges = []
    skt_cp = np.copy(skeleton)
    h, w = skeleton.shape[:2]
    if points is None:
        points, type2pts, pts2type = get_skeleton_keypts(skeleton)

    if len(points) < 1:
        return edges

    pts_set = set(points)
    while len(pts_set) > 0:
        # select max value of crossing
        cross_types = list(type2pts.keys())
        max_type = max(cross_types)

        # pop a starting point
        max_tp_pts = type2pts[max_type]
        max_tp_pts = max_tp_pts.intersection(pts_set)
        if len(max_tp_pts) == 0:
            type2pts.pop(max_type)
            continue
        (x, y) = max_tp_pts.pop()
        type2pts[max_type] = max_tp_pts
        if len(max_tp_pts) == 0:
            type2pts.pop(max_type)

        edges_cur = walk_along_edge(
            (x, y), pts_set, pts2type, skt_cp, h, w)
        edges.extend(edges_cur)
    return edges


def connect_edges(edges, dist_thresh=3):
    """The branch decomposition method sometimes makes fragments for a branch
    sine thinning process is not perfect, these segments should be connected to
    one branch to keep the branch's wholeness"""
    N = len(edges)
    if N == 0:
        return []
    sides = [[branch[0], branch[-1]] for branch in edges]
    sides_arr = np.array(sides, dtype=np.float32)
    pts = np.concatenate([sides_arr[:, 0], sides_arr[:, 1]], axis=0)
    dist_mat = get_dis_matrix(pts) + (999 * np.eye(N * 2))
    dist_mat[:N, N:] = dist_mat[:N, N:] + (999 * np.eye(N))
    dist_mat[N:, :N] = dist_mat[N:, :N] + (999 * np.eye(N))
    is_neighbor = (dist_mat < dist_thresh).astype(np.int32)
    nbrs = np.sum(is_neighbor, axis=0)
    candidates = set(np.where(nbrs == 1)[0].tolist())
    connect_ij = dict()
    ij_set = set()
    while len(candidates) > 0:
        pt_i = candidates.pop()
        pt_j = np.where(is_neighbor[pt_i])[0].tolist()
        if len(pt_j) != 1:
            continue
        pt_j = pt_j[0]
        if pt_j not in candidates:
            continue
        connect_ij[pt_i] = pt_j
        connect_ij[pt_j] = pt_i
        candidates.remove(pt_j)
        ij_set = ij_set.union({pt_i, pt_j})
    unchanged_edges = [edges[i] for i in range(N)
                        if (i not in ij_set) and ((i + N) not in ij_set)]
    new_edges = []
    while len(ij_set) > 0:
        cur_st_id = ij_set.pop()
        cur_ed_id = cur_st_id + (N if (cur_st_id < N) else -N)
        cur_edge = edges[cur_st_id] if (cur_st_id < N) else edges[cur_st_id % N][::-1]
        # go to new edge start pt
        while cur_st_id in connect_ij:
            pt_i = connect_ij[cur_st_id]
            if cur_st_id in ij_set:
                ij_set.remove(cur_st_id)
            if pt_i not in ij_set:
                break
            edge_i = edges[pt_i][::-1] if (pt_i < N) else edges[pt_i % N]
            cur_edge = edge_i + cur_edge
            ij_set.remove(pt_i)
            cur_st_id = pt_i + (N if (pt_i < N) else -N)
        # go to new edge end pt
        while cur_ed_id in connect_ij:
            pt_j = connect_ij[cur_ed_id]
            if cur_ed_id in ij_set:
                ij_set.remove(cur_ed_id)
            if pt_j not in ij_set:
                break
            edge_j = edges[pt_j] if (pt_j < N) else edges[pt_j % N][::-1]
            cur_edge = cur_edge + edge_j
            ij_set.remove(pt_j)
            cur_ed_id = pt_j + (N if (pt_j < N) else -N)
        new_edges.append(cur_edge)
    new_edges = new_edges + unchanged_edges
    return new_edges


def get_dis_matrix(pts):
    """Get distances between each two points.
    Args:
        pts, [(pt_x, pt_y), ... ]
    Return:
        distances, matrix of shape (len_pts, len_pts)
    """
    pts_len = len(pts)
    if pts_len < 1:
        return None
    if isinstance(pts, list):
        pts = np.float32(pts)
    deta_xy = np.repeat(pts[None, ...], pts_len, axis=0) - \
              np.repeat(pts[:, None, :], pts_len, axis=1)
    deta_xy = deta_xy.astype(np.float32)
    deta_xy[..., 0] += 1e-8
    z = deta_xy[..., 0] + deta_xy[..., 1] * 1j
    distances = np.abs(z)
    return distances


def dedupe_points(pts):
    new_pts = []
    distances = get_dis_matrix(pts)
    dis_index = distances < 3
    max_nbrs = np.argsort(
        np.sum(dis_index, axis=1))[::-1]
    pts_ids_set = {i for i in range(len(pts))}
    new_ids_set = set()
    for cluster_index in max_nbrs.tolist():
        if cluster_index in new_ids_set:
            continue
        new_pts.append(pts[cluster_index])
        ids = np.where(dis_index[cluster_index])[0]
        cur_set = set(ids.tolist())
        new_ids_set = new_ids_set.union(cur_set)
        pts_ids_set -= cur_set
        if len(pts_ids_set) < 1:
            break
    return new_pts


def is_junction_of_branch_sides(branches, dist_thresh=3, edpt_type=0, junc_type=1):
    """ Label the side points type of each branch in {branches}
    Args:
        branches: a list, [branch, ...], each branch is a list of points
        dist_thresh: int, to judge if two points are close enough

    Return:
        side_pts_type, array of shape (len(branches), 2), label the type
                       of all side points, 0 means endpoint, 1 means junctions
    """
    assert edpt_type != junc_type
    len_branches = len(branches)
    if len_branches < 1:
        return np.zeros(shape=(len_branches, 2), dtype=np.int32), [], []

    proj_pts = []
    for edge in branches:
        (x1, y1), (x2, y2) = edge[0], edge[-1]
        proj_pts.extend([(x1, y1), (x2, y2)])
    dist_mat = 999 * np.eye(len_branches * 2) + get_dis_matrix(proj_pts)
    side_pts_type = np.zeros(shape=(len_branches * 2,), dtype=np.int32)
    dist_mat = (dist_mat < dist_thresh).astype(np.int32)
    neighbors = np.sum(dist_mat, axis=0)
    is_endpoint = (neighbors < 1)   # endpoints have non-neighbours close enough
    is_junction = (neighbors >= 1)  # junctions have neighbours close enough
    side_pts_type[is_endpoint] = edpt_type
    side_pts_type[is_junction] = junc_type
    side_pts_type = side_pts_type.reshape((len_branches, 2)).tolist()
    proj_pts = np.int32(proj_pts)
    endpts = proj_pts[is_endpoint].tolist()
    juncpts = proj_pts[is_junction].tolist()

    endpts = dedupe_points(endpts) if len(endpts) > 1 else endpts
    juncpts = dedupe_points(juncpts) if len(juncpts) > 1 else juncpts

    return side_pts_type, endpts, juncpts


def get_dense_curves(branch_pts, clength, npt):
    len_branch = len(branch_pts)
    branch_pts = torch.as_tensor(branch_pts, dtype=torch.float32)
    if len_branch >= clength:
        if npt == 2:
            tmp = [branch_pts[: 1 - clength], branch_pts[clength - 1:]]
        elif (clength - 1) % (npt - 1) == 0: # (clength, npt): (37, 5), (33, 5)
            step = (clength - 1) // (npt - 1)
            tmp = [branch_pts[i: 1 - clength + i] for i in range(0, clength - 1, step)] + [branch_pts[clength - 1:]]
        else:
            tmp = [branch_pts[i: 1 - clength + i] for i in range(clength - 1)] + [branch_pts[clength - 1:]]
        curves = torch.stack(tmp, dim=1)
    elif len_branch >= 2:
        if npt == 2:
            curves = torch.stack([branch_pts[0], branch_pts[-1]])[None, ...]
        else:
            curves = branch_pts[None, ...]
    else:
        raise ValueError(f"branch_pts: {branch_pts}")
    return curves, branch_pts


def get_directed_curves(curves, st_type, md_type, ed_type, clength=10, stpf=0.5):
    """Sample curves of a branch, create curve label.

    Args:
       curves: tensor, with shape (N, npt, 2), come from same branch
       st_type: int, the type of start point of a branch
       md_type: int, the type of connection point of a branch except start-pt and last-pt
       ed_type: int, the type of last point of a branch
       clength: int, length of a curve
       stpf: float in (0, 1), step factor when sampling curves

    Return:
        curves, tensor of shape (M, npt, 2)
        cid, tensor of shape (M,)
    """
    step = round(clength * stpf)
    half_step = max(min(step // 2, 3), 1)
    assert step > 0 and step < clength
    num_curves, num_pts = curves.shape[:2]
    if num_curves > 2:
        id_st, id_ed = torch.arange(0, half_step), \
                       torch.arange(num_curves - half_step, num_curves)
        offset = (num_curves % step) // 2
        md_st = half_step + offset
        md_ed = max(num_curves - half_step - offset, md_st)

        id_md = torch.arange(md_st, md_ed, step)
        if len(id_md) > 0 and id_md[-1] < md_ed:
            last_md = torch.as_tensor([md_ed])
            id_md = torch.cat([id_md, last_md])

        # sampling curves
        curves_st = torch.index_select(curves, 0, id_st)
        curves_md = torch.index_select(curves, 0, id_md)
        curves_ed = torch.index_select(curves, 0, id_ed)
        inv_curves_md = torch.index_select(curves_md, 1, torch.arange(num_pts - 1, -1, -1))
        inv_curves_ed = torch.index_select(curves_ed, 1, torch.arange(num_pts - 1, -1, -1))
        curves = torch.cat([curves_st, curves_md, inv_curves_md, inv_curves_ed])
        cid_st = torch.zeros((curves_st.shape[0],), dtype=torch.long) + st_type
        cid_md = torch.zeros((curves_md.shape[0] * 2,), dtype=torch.long) + md_type
        cid_ed = torch.zeros((inv_curves_ed.shape[0],), dtype=torch.long) + ed_type
        cid = torch.cat([cid_st, cid_md, cid_ed])
    else:
        curves_st = curves[:1]
        inv_curves_ed = torch.index_select(curves[-1:], 1, torch.arange(num_pts - 1, -1, -1))
        curves = torch.cat([curves_st, inv_curves_ed])
        cid_st = torch.zeros((curves_st.shape[0],), dtype=torch.long) + st_type
        cid_ed = torch.zeros((inv_curves_ed.shape[0],), dtype=torch.long) + ed_type
        cid = torch.cat([cid_st, cid_ed])

    return curves, cid


def reshape_curves(curves, npt):
    if isinstance(curves, torch.Tensor):
        assert curves.ndim == 3
        if curves.shape[1] == npt:
            npts_curves = curves
        else:
            npts_curves = F.interpolate(
                curves[None, ...], size=(npt, 2),
                mode='bilinear', align_corners=True
            )[0]
    elif isinstance(curves, list):
        npts_curves = []
        for per_curve in curves:
            npts_curve = F.interpolate(
                torch.as_tensor(per_curve, dtype=torch.float32)[None, None, ...],
                size=(npt, 2), mode='bilinear', align_corners=True)
            npts_curves.append(npts_curve)
        npts_curves = torch.cat(npts_curves, dim=0).squeeze(dim=1)
    else:
        raise ValueError('')
    return npts_curves


def create_ann_per_branch(branch_pts, side_type, npt, rule='overlap_10_0.6'):
    """Get the real info of each image labeling without resizing or augmentations

    Args:
         branch_pts, points list, like [(x, y), ... ]
         side_type, list of two values, (first pt type, last pt type)
         npt, int, number of points to describe a curve
         rule, str, like 'rule1_3', 'rule3_7', 'rule3_13'

    Return:
        annotation, dict
    """
    st_type, ed_type = side_type
    md_type = CONNECT_PT
    branch = torch.as_tensor(branch_pts, dtype=torch.float32)
    clength = eval(rule.split('_')[1])
    if rule.split('_')[0] == "branch":
        # directly use branch pts
        inv_branch = torch.as_tensor(branch_pts[::-1], dtype=torch.float32)
        curves = torch.stack([branch, inv_branch])
        cid = torch.as_tensor([st_type, ed_type], dtype=torch.long)
    elif rule.split('_')[0] == "overlap":
        curves, _ = get_dense_curves(branch_pts, clength, npt)
        curves, cid = get_directed_curves(
            curves, st_type, md_type, ed_type, clength, eval(rule.split('_')[2]))
    else:
        raise ValueError(f"Wrong input of rule: {rule}")

    curves_num = len(curves)
    if curves_num == 0:
        return {}

    npts_curves = reshape_curves(curves, npt)

    annotation = {
        # all pts of branch
        # "branch": branch[None, ...],         # tensor,  (1, N, 2)
        # curves are split from branch according to rule
        "curves": npts_curves,               # tensor,  (curve_num, npt, 2)
        # what type of a curve's start-point: 0 - middle points, 1 - junctions, 2 - endpoints
        "cids": cid,                         # tensor,  (curve_num,)
    }

    return annotation


def decompose_skeleton(sk_masks, rule='overlap_10_0.6', npt=2, dil_iters=2):
    """
    Decompose skeleton mask and get the graph components: endpoints, junction points and lines.
    
    Args:
        sk_masks: list of skeleton masks, expected to contain a single mask with shape (h, w)
        rule: rule for curve decomposition
        npt: number of points to represent a curve
        dil_iters: number of dilation iterations
        
    Returns:
        Dictionary containing branches, keypoints, and their attributes
    """
    # Ensure sk_masks contains only one element
    assert len(sk_masks) == 1, "Function now assumes sk_masks contains only one element"
    
    h, w = sk_masks[0].shape[:2]
    size = torch.as_tensor([w, h], dtype=torch.float32)
    branches = []
    keypoint_directions = []
    
    # Process the single skeleton mask
    graph_mask = sk_masks[0]
    
    # Preprocess skeleton
    sk_mask = (graph_mask > 0).astype(np.uint8)
    sk_mask = cv2.dilate(sk_mask, kernel=DIL_KERNER, iterations=dil_iters)
    sk_mask = morphology.skeletonize(sk_mask, method='lee').astype(np.uint8)
    
    # Extract branches and keypoints
    branches_i = split_skeleton(sk_mask)
    side_pts_type, endpts, juncpts = is_junction_of_branch_sides(
        branches_i, edpt_type=END_PT, junc_type=JUNCTION_PT)
    
    # Store branch data for direction calculation
    branch_data = {}
    
    # Process each branch
    for bid, branch_pts in enumerate(branches_i):
        # Store branch data
        branch_data[bid] = {
            'points': branch_pts,
            'start': tuple(branch_pts[0]),
            'end': tuple(branch_pts[-1]),
            'start_type': side_pts_type[bid][0],
            'end_type': side_pts_type[bid][1]
        }
        
        # Process branch for curves
        branch_ann = create_ann_per_branch(branch_pts, side_pts_type[bid], npt, rule=rule)
        if len(branch_ann) == 0:
            continue
            
        M = len(branch_ann['curves'])
        branches.append({
            "branch_id": bid,
            "curves": torch.as_tensor(branch_ann['curves']) / size,
            "cids": torch.as_tensor(branch_ann['cids']),
            "clabels": torch.zeros(size=(M,), dtype=torch.long)
        })
    
    # Initialize keypoint to branch mapping
    keypoint_to_branches = {}
    
    # Map endpoints to branches and mark branches as assigned
    assigned_branches = set()
    for endpoint in endpts:
        endpoint_tuple = tuple(endpoint)
        keypoint_to_branches[endpoint_tuple] = []
        
        for branch_id, data in branch_data.items():
            # Check if branch connects to this endpoint
            if (data['start_type'] == END_PT and np.linalg.norm(np.array(endpoint_tuple) - np.array(data['start'])) < 3) or \
               (data['end_type'] == END_PT and np.linalg.norm(np.array(endpoint_tuple) - np.array(data['end'])) < 3):
                keypoint_to_branches[endpoint_tuple].append(branch_id)
                assigned_branches.add(branch_id)
    
    # Process junction points iteratively
    remaining_branches = set(branch_data.keys()) - assigned_branches
    unassigned_junctions = set(tuple(j) for j in juncpts)
    
    # Continue iterating until no more junctions can be assigned as "new endpoints"
    while unassigned_junctions and remaining_branches:
        # Find junctions that act as "new endpoints" for remaining branches
        newly_assigned = False
        
        # Store temporary mappings for this iteration
        junction_connections = {}
        
        # Check each unassigned junction against remaining branches
        for junction in list(unassigned_junctions):
            junction_connections[junction] = []
            
            for branch_id in remaining_branches:
                data = branch_data[branch_id]
                if (data['start_type'] == JUNCTION_PT and np.linalg.norm(np.array(junction) - np.array(data['start'])) < 3) or \
                   (data['end_type'] == JUNCTION_PT and np.linalg.norm(np.array(junction) - np.array(data['end'])) < 3):
                    junction_connections[junction].append(branch_id)
            
            # If junction connects to exactly one remaining branch, assign it as a "new endpoint"
            if len(junction_connections[junction]) == 1:
                branch_id = junction_connections[junction][0]
                keypoint_to_branches[junction] = [branch_id]
                assigned_branches.add(branch_id)
                remaining_branches.remove(branch_id)
                unassigned_junctions.remove(junction)
                newly_assigned = True
        
        # If no new assignments were made, break the loop
        if not newly_assigned:
            break
    
    for junction in list(unassigned_junctions):
            keypoint_to_branches[junction] = [-1]
    
    # Calculate directions from keypoints to branches
    all_keypoints = endpts + juncpts
    _key_pts = torch.as_tensor(all_keypoints, dtype=torch.float32) / size
    
    for keypoint in all_keypoints:
        keypoint_tuple = tuple(keypoint)
        keypoint_array = np.array(keypoint_tuple)
        connected_branches = keypoint_to_branches.get(keypoint_tuple, [])
        
        # Filter out -1 branch IDs for direction calculation
        valid_branches = [bid for bid in connected_branches if bid != -1]
        
        if valid_branches:
            # Calculate directions to connected branches
            directions = []
            for branch_id in valid_branches:
                branch_points = np.array(branch_data[branch_id]['points'])
                # filter branch points within 10 pixels of keypoint
                branch_points = branch_points[np.linalg.norm(branch_points - keypoint_array, axis=1) < 20]
                # Calculate mean point of branch
                mean_point = np.mean(branch_points, axis=0)
                
                # Calculate direction from keypoint to mean point
                direction = mean_point - keypoint_array
                norm = np.linalg.norm(direction)
                
                if norm > 0:
                    direction = direction / norm  # Normalize
                else:
                    direction = np.array([0.0, 0.0])
                
                directions.append(direction)
            
            # Average the directions if there are multiple
            if directions:
                avg_direction = np.mean(directions, axis=0)
                norm = np.linalg.norm(avg_direction)
                if norm > 0:
                    avg_direction = avg_direction / norm
                keypoint_directions.append(avg_direction)
            else:
                keypoint_directions.append(np.array([0.0, 0.0]))
        else:
            # No valid connected branches
            keypoint_directions.append(np.array([0.0, 0.0]))
    
    # Prepare keypoint labels
    _plabels = [END_PT] * len(endpts) + [JUNCTION_PT] * len(juncpts)
    _plabels = -1 + torch.as_tensor(_plabels, dtype=torch.long)
    
    # Convert keypoint directions to tensor
    keypoint_directions = np.array(keypoint_directions, dtype=np.float32)
    keypoint_directions = torch.from_numpy(keypoint_directions)
    
    # Assemble final target dictionary
    target = {
        "branches": branches,
        "key_pts": _key_pts,
        "plabels": _plabels,
        "keypoint_directions": keypoint_directions,
    }
    
    return target