import random
import os
import yaml
import wandb
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from datasets.skeleton import build_dataset, visualize_target
from models.deformable_detr import build_model
from criterion.loss import build_criterion
from datasets.reconstruct_graph import PostProcess
from datasets.data_prefetcher import collate_fn
from torch.utils.data import DataLoader
from criterion.skeleton_eval import SkeletonEvaluator
import cv2
import argparse


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, model, criterion, device, config):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.config = config
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.postprocessor = PostProcess(eval_score=None)

        # Setup optimizer
        def match_name_keywords(n, keywords):
            return any(kw in n for kw in keywords)

        lr_backbone_names = config.get("lr_backbone_names", [])
        lr_linear_proj_names = config.get("lr_linear_proj_names", [])

        param_dicts = [
            {
                "params": [p for n, p in model.named_parameters() if not match_name_keywords(n, lr_backbone_names) and not match_name_keywords(n, lr_linear_proj_names) and p.requires_grad],
                "lr": config.get("lr", 1e-4),
            },
            {
                "params": [p for n, p in model.named_parameters() if match_name_keywords(n, lr_backbone_names) and p.requires_grad],
                "lr": config.get("lr_backbone", 1e-5),
            },
            {
                "params": [p for n, p in model.named_parameters() if match_name_keywords(n, lr_linear_proj_names) and p.requires_grad],
                "lr": config.get("lr", 1e-4) * config.get("lr_linear_proj_mult", 1.0),
            }
        ]

        self.optimizer = torch.optim.AdamW(
            param_dicts,
            lr=config.get("lr", 1e-4),
            weight_decay=config.get("weight_decay", 1e-4)
        )

        # Setup data loaders
        train_dataset = build_dataset(is_train=True, config=config)
        test_dataset = build_dataset(is_train=False, config=config)
        self.train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                                       collate_fn=collate_fn, num_workers=config["num_workers"], pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                      collate_fn=collate_fn, num_workers=config["num_workers"], pin_memory=True)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config["t_0"],
            T_mult=config["t_mult"],
            eta_min=config["lr"] * 0.01
        )

    def train_one_epoch(self, use_wandb):
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, disable=use_wandb)
        for images, targets in progress_bar:
            images = images.to(self.device)
            targets = [{k: v.to(self.device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            progress_bar.set_description(f"Loss: {loss.item():.4f}")
            loss.backward()
            if self.config["clip_max_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["clip_max_norm"])
            self.optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(self.train_loader)
        self.lr_scheduler.step()
        return avg_loss

    def validate(self, use_wandb):
        self.model.eval()
        eval_score = self.config["eval_score"]
        sk_evaluator = SkeletonEvaluator()
        val_images = []
        with torch.no_grad():
            for imgs, targets in tqdm(self.test_loader, disable=use_wandb):
                target = targets[0]
                w, h = target['orig_size'].data.cpu().numpy()
                data_idx = int(target['id'].cpu().numpy())
                inputName, targetName = self.test_loader.dataset.id2name(data_idx)
                gt_skeleton = (cv2.imread(targetName, 0) > 0).astype(np.uint8) * 255

                imgs = imgs.to(self.device)
                outputs = self.model(imgs)
                targets = [{k: v.to(self.device) for k, v in target.items()}]
                orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
                results_dict = self.postprocessor(outputs, orig_target_sizes, ignore_graph=True)

                batch_size = imgs.tensors.shape[0]
                assert batch_size == 1

                pred = results_dict['curves'][0]
                ptspred = results_dict['keypoints'][0]

                _, pred_mask = self.postprocessor.visualise_curves(pred, ptspred, eval_score, np.zeros((h, w, 3), dtype=np.uint8))
                sk_evaluator.update([(gt_skeleton, pred_mask, inputName)])

                # Append only the first 4 visualizations.
                if len(val_images) < 4:
                    raw_img = Image.open(inputName).convert("RGB")
                    _raw_img = np.array(raw_img)[:, :, ::-1]
                    vis_img = np.copy(_raw_img)
                    vis_img, curves_mask = self.postprocessor.visualise_curves(pred, ptspred, eval_score, vis_img, thinning=True, ch3mask=True, vmask=255)
                    pred_vis, pts_mask = self.postprocessor.visualise_pts(ptspred, eval_score, curves_mask)
                    
                    gt_vis = visualize_target(target)
                    gt_vis = cv2.resize(gt_vis, (vis_img.shape[1], vis_img.shape[0]))
                    
                    # Concatenate original visualization with ground truth skeleton
                    vis_combined = np.concatenate((vis_img, pred_vis, gt_vis), axis=1).astype(np.uint8)
                    vis_combined = cv2.cvtColor(vis_combined, cv2.COLOR_BGR2RGB)
                    val_images.append(wandb.Image(vis_combined, caption=f"{inputName}"))

        metrics = sk_evaluator.summarize(score_threshold=eval_score, offset_threshold=0.01)
        results = {
            "val_images": val_images,
            "val_f1_cnt": metrics.get('cnt_f1', 0),
            "val_f1_m": metrics.get('m_f1', 0),
        }
        return results

    def save_checkpoint(self, epoch):
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'epoch': epoch,
            'config': self.config,
        }
        wandb_name = self.config["wandb"]["name"]
        torch.save(checkpoint, os.path.join(self.output_dir, f'{wandb_name}_checkpoint_{epoch}.pth'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    args = parser.parse_args()
    config_path = args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Initialize wandb
    wandb.init(project=config["wandb"]["project"], name=config["wandb"]["name"],
               dir=config["wandb"]["dir"], config=config)
    wandb.config.update(config)
    use_wandb = os.getenv("WANDB_MODE") != "disabled"
    # set_seed(config["seed"])
    device = torch.device("cuda")

    model = build_model(config, device)
    # Load checkpoint
    if config["load_path"] and config["load_path"] != "pretrain":
        checkpoint = torch.load(config["load_path"])
        model.load_state_dict(checkpoint["model"])
        print(f"Loaded checkpoint from {config['load_path']}.")

    criterion = build_criterion(config, device)

    trainer = Trainer(model, criterion, device, config)
    for epoch in range(config["epochs"]):
        train_loss = trainer.train_one_epoch(use_wandb)
        wandb.log({"train_loss": train_loss}, step=epoch)
        print(f"Trained [{epoch}/{config['epochs']}] epoches, loss: {train_loss:.4f}")

        if epoch % config["eval_epochs"] == 0 or epoch == config["epochs"] - 1:
            print(f"Validating at [{epoch}/{config['epochs']}] epoches...")
            results = trainer.validate(use_wandb)
            wandb.log(results, step=epoch)
            print("f1 = ", results.get('val_f1_m', 0))
            print("cnt_f1 = ", results.get('val_f1_cnt', 0))

    trainer.save_checkpoint(config["epochs"])
    


if __name__ == "__main__":
    main()
