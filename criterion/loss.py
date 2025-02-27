import torch
import torch.nn.functional as F
from torch import nn
from .matcher import build_matcher

import torch
import torch.nn as nn
import torch.nn.functional as F


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, config):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            config: config file
        """
        super().__init__()
        self.matcher = matcher
        self.loss_class_weight = config["loss_class_weight"]
        self.loss_regression_weight = config["loss_regression_weight"]
        self.loss_keypoint_weight = config["loss_keypoint_weight"]
        self.loss_curve_weight = config["loss_curve_weight"]
        self.focal_alpha = config["focal_alpha"]

    def sigmoid_focal_loss(self, inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
        Returns:
            Loss tensor
        """
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean(1).sum() / num_boxes

    def loss_class(self, src_logits, target_labels, indices, num_boxes):
        """Classification loss (NLL)
        target_labels: tensor containing target class labels
        """
        non_tgt_label = src_logits.shape[-1] - 1
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(target_labels, indices)])
        target_classes = torch.full(src_logits.shape[:2], non_tgt_label, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = self.sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        return loss_ce

    def loss_regression(self, src_boxes, target_boxes, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           target_boxes: tensor containing target box coordinates
        """
        idx = self._get_src_permutation_idx(indices)
        src_boxes = src_boxes[idx]
        target_boxes_cat = torch.cat([t[i] for t, (_, i) in zip(target_boxes, indices)], dim=0).flatten(1)
        loss_bbox = F.l1_loss(src_boxes, target_boxes_cat, reduction='mean') * 4
        return loss_bbox

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward(self, outputs, targets):
        loss = {}
        pred_curve_class = outputs["curves"]["pred_logits"]
        pred_curve_points = outputs["curves"]["pred_points"]
        gt_curve_class = [t["clabels"] for t in targets]
        gt_curve_points = [t["curves"] for t in targets]
        curve_loss = self.forward_curves(pred_curve_class, pred_curve_points, gt_curve_class, gt_curve_points)

        pred_keypoint_class = outputs['keypoints']['pred_logits']
        pred_keypoint_points = outputs['keypoints']['pred_points']
        gt_keypoint_class = [t["plabels"] for t in targets]
        gt_keypoint_points = [t["key_pts"] for t in targets]
        keypoint_loss = self.forward_keypoints(pred_keypoint_class, pred_keypoint_points, gt_keypoint_class, gt_keypoint_points)
        loss = self.loss_curve_weight * curve_loss + self.loss_keypoint_weight * keypoint_loss
        return loss

    def forward_curves(self, pred_class, pred_points, gt_class, gt_points):
        """Compute loss for curve predictions."""
        device = pred_points.device

        # Compute the average number of target boxes
        num_boxes = sum(len(labels) for labels in gt_class)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        num_boxes = torch.clamp(num_boxes / 1, min=1).item()

        # Get matching indices
        indices = self.matcher(pred_class, pred_points, gt_class, gt_points)

        class_loss = self.loss_class(pred_class, gt_class, indices, num_boxes)
        regression_loss = self.loss_regression(pred_points, gt_points, indices)

        loss = self.loss_class_weight * class_loss + self.loss_regression_weight * regression_loss
        return loss

    def forward_keypoints(self, pred_class, pred_points, gt_class, gt_points):
        """Compute loss for keypoint predictions."""
        device = pred_points.device
        

        indices = self.matcher(pred_class, pred_points, gt_class, gt_points)

        num_boxes = sum(len(labels) for labels in gt_class)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        num_boxes = torch.clamp(num_boxes / 1, min=1).item()

        class_loss = self.loss_class(pred_class, gt_class, indices, num_boxes)
        regression_loss = self.loss_regression(pred_points, gt_points, indices)

        loss = self.loss_class_weight * class_loss + self.loss_regression_weight * regression_loss
        return loss



def build_criterion(config, device):
    matcher = build_matcher(config)

    criterion = SetCriterion(
        matcher, config,
    )
    criterion.to(device)

    return criterion
