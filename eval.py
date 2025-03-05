import os
import yaml
import torch
import numpy as np
import cv2
import argparse
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from datasets.skeleton import build_dataset
from models.deformable_detr import build_model
from datasets.reconstruct_graph import PostProcess
from datasets.data_prefetcher import collate_fn
from torch.utils.data import DataLoader
from criterion.skeleton_eval import SkeletonEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation script for BlumNetGraph')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to config file')
    parser.add_argument('--load_path', type=str, help='Path to checkpoint file to load (overrides config)')
    parser.add_argument('--max_vis', type=int, default=20, help='Maximum number of visualizations to save')
    return parser.parse_args()


def evaluate(config, load_path, max_vis=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    model = build_model(config, device)

    # Load checkpoint
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Create output directory
    output_dir = Path(config["wandb"]["dir"], config["wandb"]["name"])
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    # Setup dataset and dataloader
    test_dataset = build_dataset(is_train=False, config=config)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             collate_fn=collate_fn, num_workers=config["num_workers"], pin_memory=True)

    # Setup evaluator and postprocessor
    sk_evaluator = SkeletonEvaluator()
    postprocessor = PostProcess(eval_score=None)
    score_threshold = config["eval_score"]
    offset_threshold = 0.01  # Default offset threshold for evaluation

    print(f"Evaluating model from {load_path}...")
    print(f"Saving results to {output_dir}")
    print(f"Will save visualizations for up to {max_vis} samples with lowest F1 scores")

    # Run evaluation
    sample_results = []

    with torch.no_grad():
        for idx, (imgs, targets) in enumerate(tqdm(test_loader)):
            target = targets[0]
            w, h = target['orig_size'].data.cpu().numpy()
            data_idx = int(target['id'].cpu().numpy())
            inputName, targetName = test_loader.dataset.id2name(data_idx)
            gt_skeleton = (cv2.imread(targetName, 0) > 0).astype(np.uint8) * 255

            imgs = imgs.to(device)
            outputs = model(imgs)
            targets = [{k: v.to(device) for k, v in target.items()}]
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results_dict = postprocessor(outputs, orig_target_sizes, ignore_graph=True)

            pred = results_dict['curves'][0]
            ptspred = results_dict['keypoints'][0]

            _, pred_mask = postprocessor.visualise_curves(pred, score_threshold, np.zeros((h, w, 3), dtype=np.uint8))

            # Evaluate the current sample
            sample_metrics = sk_evaluator.evaluate_sample(gt_skeleton, pred_mask, offset_threshold)
            
            # Store all information needed for visualization
            sample_results.append({
                'inputName': inputName,
                'targetName': targetName,
                'f1': sample_metrics['f1'],
                'precision': sample_metrics['precision'],
                'recall': sample_metrics['recall'],
                'pred': pred,
                'ptspred': ptspred,
                'w': w,
                'h': h
            })

            # Add to evaluator for cumulative metrics
            sk_evaluator.update([(gt_skeleton, pred_mask, inputName)])

    # Sort results by F1 score (ascending)
    sample_results.sort(key=lambda x: x['f1'])
    
    # Take the lowest max_vis samples
    worst_samples = sample_results[:max_vis]
    
    # Generate visualizations for the worst samples
    print(f"\nGenerating visualizations for {len(worst_samples)} samples with lowest F1 scores...")
    
    for i, sample in enumerate(worst_samples):
        # Get sample data
        inputName = sample['inputName']
        targetName = sample['targetName']
        pred = sample['pred']
        ptspred = sample['ptspred']
        f1_score = sample['f1']
        precision = sample['precision']
        recall = sample['recall']
        
        # Create visualization
        raw_img = Image.open(inputName).convert("RGB")
        _raw_img = np.array(raw_img)[:, :, ::-1]  # Convert to BGR for cv2
        vis_img = np.copy(_raw_img)
        curve_score = 0.5
        pts_score = 0.8
        vis_img, curves_mask = postprocessor.visualise_curves(pred, curve_score, vis_img, thinning=True, ch3mask=True, vmask=255)
        vis_img, pts_mask = postprocessor.visualise_pts(ptspred, pts_score, vis_img)

        gt_vis = cv2.imread(targetName, 0)
        gt_vis[gt_vis > 0] = 255
        gt_vis = np.stack([gt_vis]*3, axis=2)
        gt_vis = cv2.morphologyEx(gt_vis, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

        # Combine visualizations
        vis_combined = np.concatenate((vis_img, curves_mask, gt_vis), axis=1).astype(np.uint8)

                # Add F1 score and ranking to the image
        text = f"F1: {f1_score:.4f} (P: {precision:.4f}, R: {recall:.4f}) - Rank: {i+1}/{len(worst_samples)}"
        cv2.putText(vis_combined, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

        # Save visualization
        img_basename = os.path.basename(inputName)
        output_path = vis_dir / f"{i+1:03d}_f1_{f1_score:.4f}_{img_basename}.png"
        cv2.imwrite(str(output_path), vis_combined)

    # Calculate and print overall metrics
    print("\nCalculating final metrics...")
    metrics = sk_evaluator.summarize(score_threshold=score_threshold, offset_threshold=offset_threshold)

    print("\nEvaluation Results:")
    print(f"Mean F1 score: {metrics.get('m_f1', 0):.4f}")
    print(f"Cumulative F1: {metrics.get('cnt_f1', 0):.4f}")
    print(f"Precision: {metrics.get('cnt_precision', 0):.4f}")
    print(f"Recall: {metrics.get('cnt_recall', 0):.4f}")
    
    return metrics


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override load_path if provided via command line
    load_path = args.load_path if args.load_path else config["load_path"]

    evaluate(config, load_path, args.max_vis)
