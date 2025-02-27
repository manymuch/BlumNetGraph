import torch
import torch.nn.functional as F
from torch import nn
from .matcher import build_matcher

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_input(t, requires_grad=False, device=torch.device('cpu')):
    """Make zero inputs for AE loss.

    Args:
        t (torch.Tensor): input
        requires_grad (bool): Option to use requires_grad.
        device: torch device

    Returns:
        torch.Tensor: zero input.
    """
    inp = torch.autograd.Variable(t, requires_grad=requires_grad)
    inp = inp.sum()
    inp = inp.to(device)
    return inp


class AELoss(nn.Module):
    """Associative Embedding loss. This loss is recommended for most grouping 
    including grouping graph id in Blumnet.

    `Associative Embedding: End-to-End Learning for Joint Detection and
    Grouping <https://arxiv.org/abs/1611.05424v2>`
    """

    def __init__(self, loss_type='exp', push_loss_factor=1, pull_loss_factor=1):
        super().__init__()
        assert loss_type in ['exp', 'max']
        self.loss_type = loss_type
        self.push_loss_factor = push_loss_factor
        self.pull_loss_factor = pull_loss_factor

    def singleTagLoss(self, pred_tag, gt_tag):
        """Associative embedding loss for one image.

        Args:
            pred_tag (torch.Tensor[N,]): tag channels of output.
            gt_tags (torch.Tensor[N,]): tag channels of gt.
        """
        gt_tag = gt_tag.int()
        max_bid = torch.max(gt_tag).cpu().data.numpy()
        tags = []
        pull = 0
        for per_bid in range(-1, max_bid + 1):
            same_tag = pred_tag[gt_tag == per_bid]
            if len(same_tag) == 0:
                continue
            tags.append(torch.mean(same_tag, dim=0))
            pull = pull + torch.mean((same_tag - tags[-1].expand_as(same_tag))**2)

        num_tags = len(tags)
        if num_tags == 0:
            return (
                _make_input(torch.zeros(1).float(), device=pred_tag.device),
                _make_input(torch.zeros(1).float(), device=pred_tag.device))
        elif num_tags == 1:
            return (_make_input(
                torch.zeros(1).float(), device=pred_tag.device), pull)

        tags = torch.stack(tags)

        size = (num_tags, num_tags)
        A = tags.expand(*size)
        B = A.permute(1, 0)

        diff = A - B

        if self.loss_type == 'exp':
            diff = torch.pow(diff, 2)
            push = torch.exp(-diff)
            push = torch.sum(push) - num_tags
        elif self.loss_type == 'max':
            diff = 1 - torch.abs(diff)
            push = torch.clamp(diff, min=0).sum() - num_tags
        else:
            raise ValueError('Unknown ae loss type')

        push_loss = push / ((num_tags - 1) * num_tags) * 0.5
        pull_loss = pull / (num_tags)

        return push_loss, pull_loss

    def forward(self, tags, gt_tags):
        """Accumulate the tag loss for each image in the batch.

        Note:
            batch_size: B

        Args:
            pred_tags (torch.Tensor[BxN]): tag channels of output.
            gt_tags (torch.Tensor[BxN]): tag channels of gt.
        """
        pushes, pulls = [], []
        batch_size = tags.size(0)
        for i in range(batch_size):
            push, pull = self.singleTagLoss(tags[i], gt_tags[i])
            pushes.append(push)
            pulls.append(pull)

        return (torch.stack(pushes) * self.push_loss_factor,
                torch.stack(pulls) * self.pull_loss_factor)


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
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



class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, weight_dict, losses, gt_pts_key, gt_pts_label, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.gt_pts_key = gt_pts_key
        self.gt_pts_label = gt_pts_label

    def loss_labels(self, outputs, targets, indices, num_boxes, gt_pts_key, gt_pts_label):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        non_tgt_label = src_logits.shape[-1] - 1
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t[gt_pts_label][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], non_tgt_label, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, gt_pts_key, gt_pts_label):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t[gt_pts_key][i] for t, (_, i) in zip(targets, indices)], dim=0).flatten(1)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='mean') * 4
        losses = {'loss_bbox': loss_bbox}
        return losses

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

    def get_loss(self, loss, outputs, targets, indices, num_boxes, gt_pts_key, gt_pts_label, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, gt_pts_key, gt_pts_label, **kwargs)

    def forward(self, outputs, targets):
        loss = {}
        if 'curves' in outputs:
            gt_pts_key, gt_pts_label = self.gt_pts_key, self.gt_pts_label
            per_loss = self._forward(outputs['curves'], targets, gt_pts_key, gt_pts_label)
            loss.update(
                {f"c{k}": v for k, v in per_loss.items()}
            )
        if 'pts' in outputs:
            per_loss = self._forward(outputs['pts'], targets, gt_pts_key='key_pts', gt_pts_label='plabels')
            loss.update(
                {f"p{k}": v for k, v in per_loss.items()}
            )
        return loss

    def _forward(self, outputs, targets, gt_pts_key, gt_pts_label):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        device = outputs['pred_boxes'].device
        npt = outputs['pred_boxes'].shape[-1] // 2

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t[gt_pts_label]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        num_boxes = torch.clamp(num_boxes / 1, min=1).item()

        losses = {}

        if npt > 5:
            pt_ids = torch.arange(npt, dtype=torch.long, device=device)
        else:
            pt_ids = torch.as_tensor([0, npt - 1], dtype=torch.long, device=device)
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        indices = self.matcher(outputs_without_aux, targets, gt_pts_key, gt_pts_label, pt_ids=pt_ids)
        l_dict = {}
        for loss in self.losses:
            kwargs = {}
            l_dict.update(self.get_loss(
                loss, outputs_without_aux, targets, indices, num_boxes, gt_pts_key, gt_pts_label, **kwargs))
        losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets, gt_pts_key, gt_pts_label, pt_ids=pt_ids)
                l_dict = {}
                for loss in self.losses:
                    kwargs = {} if (loss != 'labels') else {'log': True}
                    l_dict.update(self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes, gt_pts_key, gt_pts_label, **kwargs))
                losses.update({f"{k}_{i}": v for k, v in l_dict.items()})

        return losses


def build_criterion(config, device):
    matcher = build_matcher(config)
    weight_dict = {
        'loss_ce': config.get("cls_loss_coef", 1),
        'loss_bbox': config.get("bbox_loss_coef", 5),
    }
    add_items = {}
    for k, v in weight_dict.items():
        for prefix in ['c', 'p']:
            if prefix == 'c':
                add_items[f"{prefix}{k}"] = v
            else:
                relative_v = v / 10  # among all queries, only about 10% is for points
                add_items[f"{prefix}{k}"] = config.get("pts_loss_coef", 1) * relative_v
    weight_dict.update(add_items)

    if config.get("aux_loss", False):
        aux_weight_dict = {}
        dec_layers = config.get("dec_layers", 6)
        for i in range(dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(
        matcher, weight_dict,
        losses=['labels', 'boxes'],
        gt_pts_key="curves",
        gt_pts_label='clabels',
        focal_alpha=config.get("focal_alpha", 0.25),
    )
    criterion.to(device)

    return criterion
