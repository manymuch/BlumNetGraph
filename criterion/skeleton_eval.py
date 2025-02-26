# ------------------------------------------------------------------------
# Blumnet
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import cv2
import os
import numpy as np
from skimage import morphology


class SkeletonEvaluator(object):

    def __init__(self):
        self.predictions = []

    def get_f1(self, precision, recall):
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        return f1

    def update(self, predictions):
        """
        Args:
            predictions: [(gt_skeleton, pred_mask, im_path), ... ], each item is np.uint8 array
        """
        self.predictions.extend(predictions)

    def evaluate_sample(self, gt_target, pred, offset_threshold=0.01):
        """
        Evaluate a single sample (gt_target and pred pair)

        Args:
            gt_target: ground truth skeleton mask (np.uint8 array)
            pred: predicted mask (np.uint8 array)
            offset_threshold: threshold for distance matching

        Returns:
            dict: evaluation metrics for the single sample
        """
        pred_mask = morphology.skeletonize(pred, method='lee').astype(np.uint8)
        h, w = gt_target.shape
        assert (gt_target.shape == pred.shape)

        pred_yy, pred_xx = np.where(pred_mask)
        gt_yy, gt_xx = np.where(gt_target > 0)

        if len(pred_yy) < 1:
            precision, recall, f1 = 0, 0, 0
            cntR, cntP, sumR, sumP = 0, 0, len(gt_yy), 0
        else:
            pd_pts = np.stack([pred_xx, pred_yy], axis=1)
            gt_pts = np.stack([gt_xx, gt_yy], axis=1) if len(gt_yy) > 0 else np.empty((0, 2))

            pd_num = pd_pts.shape[0]
            gt_num = gt_pts.shape[0]

            offset_ths = offset_threshold * ((h**2 + w**2)**0.5)

            if gt_num == 0:
                precision, recall, f1 = 0, 1.0, 0
                cntR, cntP, sumR, sumP = 0, 0, 0, pd_num
            else:
                distances = np.linalg.norm(
                    np.repeat(pd_pts[:, None, :], repeats=gt_num, axis=1) -
                    np.repeat(gt_pts[None, :, :], repeats=pd_num, axis=0),
                    axis=2)

                cntR = np.sum(np.min(distances, axis=0) < offset_ths)
                cntP = np.sum(np.min(distances, axis=1) < offset_ths)
                sumR = gt_num
                sumP = pd_num

                precision = cntP / sumP
                recall = cntR / sumR
                f1 = self.get_f1(precision, recall)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cntR': cntR,
            'sumR': sumR,
            'cntP': cntP,
            'sumP': sumP,
            'pred_mask': pred_mask  # Return the skeletonized mask for visualization
        }

    def summarize(self, score_threshold=0.5, offset_threshold=0.01, visual_dir=None):
        """
        Cumulative evaluation across all samples

        Args:
            score_threshold: threshold for prediction scores
            offset_threshold: threshold for distance matching
            visual_dir: directory to save visualizations

        Returns:
            dict: evaluation metrics
        """
        metrics = {
            'precision': [], 'recall': [], 'f1': [], 'score_threshold': score_threshold,
            'cntR_total': 0, 'sumR_total': 0, 'cntP_total': 0, 'sumP_total': 0
        }

        for gt_target, pred, im_path in self.predictions:
            # Evaluate each sample
            sample_metrics = self.evaluate_sample(gt_target, pred, offset_threshold)

            # Store sample-level metrics
            metrics['precision'].append(sample_metrics['precision'])
            metrics['recall'].append(sample_metrics['recall'])
            metrics['f1'].append(sample_metrics['f1'])

            # Accumulate cumulative metrics
            metrics['cntR_total'] += sample_metrics['cntR']
            metrics['sumR_total'] += sample_metrics['sumR']
            metrics['cntP_total'] += sample_metrics['cntP']
            metrics['sumP_total'] += sample_metrics['sumP']

            # Visualization if needed
            if visual_dir is not None and os.path.isdir(visual_dir):
                fileid = os.path.basename(im_path).split('.')[0]
                cv2.imwrite(os.path.join(visual_dir, f"{fileid}.png"), sample_metrics['pred_mask'])

        # Calculate mean metrics across samples
        metrics['m_precision'] = round(np.mean(metrics['precision']), 4)
        metrics['m_recall'] = round(np.mean(metrics['recall']), 4)
        metrics['m_f1'] = round(np.mean(metrics['f1']), 4)

        # Calculate cumulative metrics across all samples
        metrics['cnt_precision'] = round(metrics['cntP_total'] / (metrics['sumP_total'] + (metrics['sumP_total'] == 0)), 4)
        metrics['cnt_recall'] = round(metrics['cntR_total'] / (metrics['sumR_total'] + (metrics['sumR_total'] == 0)), 4)
        metrics['cnt_f1'] = round(self.get_f1(metrics['cnt_precision'], metrics['cnt_recall']), 4)

        return metrics
