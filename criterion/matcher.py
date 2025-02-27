# ------------------------------------------------------------------------
# Blumnet
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 class_weight: float = 1,
                 regression_weight: float = 1,
                 focal_ce_cost: bool = False):
        super().__init__()
        self.class_weight = class_weight
        self.regression_weight = regression_weight
        assert class_weight != 0 or regression_weight != 0, "weights must be non-zero"
        self.focal_ce_cost = focal_ce_cost
        if focal_ce_cost:
            self.alpha = 0.25
            self.gamma = 2.0

    def forward(self, pred_class, pred_points, gt_class, gt_points):
        """ Performs the matching

        Params:
            pred_class: Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
            pred_points: Tensor of dim [batch_size, num_queries, 2] with the predicted skeleton coordinates
            gt_class: List of tensors of dim [num_target_boxes] containing the class labels
            gt_points: List of tensors of dim [num_target_boxes, 2] containing the target skeleton coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = pred_class.shape[:2]
            # We flatten to compute the cost matrices in a batch
            out_prob = pred_class.flatten(0, 1).sigmoid()
            out_pts = pred_points.flatten(0, 1)  # [batch_size * num_queries, 2]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat(gt_class)
            tgt_pts = torch.cat(gt_points).flatten(1)
            
            # Compute the classification cost.
            if self.focal_ce_cost:
                neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + 1e-8).log())
                pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
                cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
            else:
                cost_class = -out_prob[:, tgt_ids]

            # Compute the L1 cost between points
            l1_cost = torch.cdist(out_pts, tgt_pts, p=1)
            cost_regression = 4 / tgt_pts.shape[-1] * l1_cost

            # Final cost matrix
            C = self.regression_weight * cost_regression + self.class_weight * cost_class
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(pts) for pts in gt_points]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(config):
    class_weight = config["matcher_class_weight"]
    regression_weight = config["matcher_regression_weight"]
    focal_ce_cost = config["focal_ce_cost"]
    return HungarianMatcher(class_weight=class_weight,
                            regression_weight=regression_weight,
                            focal_ce_cost=focal_ce_cost)
