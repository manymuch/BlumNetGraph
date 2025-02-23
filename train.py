import random
import os
import yaml
import wandb
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import lib.misc as utils
from datasets import build_dataset
from models import build_model
from reconstruction import PostProcess
from torch.utils.data import DataLoader
from lib.libmetric import SkeletonEvaluator
import cv2


def load_config(config_path="config/default.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dataloaders(config):
    train_dataset = build_dataset(image_set='train', config=config)
    test_dataset = build_dataset(image_set='test', config=config)
    train_loader = DataLoader(train_dataset, batch_size=config.get("batch_size", 4), shuffle=True,
                              collate_fn=utils.collate_fn, num_workers=config.get("num_workers", 4), pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             collate_fn=utils.collate_fn, num_workers=config.get("num_workers", 4), pin_memory=True)
    return train_loader, test_loader


def train_one_epoch(model, criterion, data_loader, optimizer, device, clip_max_norm):
    model.train()
    total_loss = 0.0
    for images, targets in tqdm(data_loader, disable=True):
        images = images.to(device)
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        outputs = model(images)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    return avg_loss


def validate(model, test_loader, device, epoch, config):
    model.eval()
    postprocessor = PostProcess(eval_score=None)
    score_threshold = config["eval_score"]
    sk_evaluator = SkeletonEvaluator()
    val_images = []
    with torch.no_grad():
        for imgs, targets in tqdm(test_loader, disable=True):
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

            batch_size = imgs.tensors.shape[0]
            assert batch_size == 1

            pred = results_dict['curves'][0]
            ptspred = results_dict['pts'][0]

            _, pred_mask = postprocessor.visualise_curves(pred, 0.65, np.zeros((h, w, 3), dtype=np.uint8))
            sk_evaluator.update([(gt_skeleton, pred_mask, inputName)])

            # Append only the first 4 visualizations.
            if len(val_images) < 4:
                raw_img = Image.open(inputName).convert("RGB")
                _raw_img = np.array(raw_img)[:, :, ::-1]
                vis_img = np.copy(_raw_img)
                vis_img, curves_mask = postprocessor.visualise_curves(pred, 0.65, vis_img, thinning=True, ch3mask=True, vmask=255)
                vis_img, pts_mask = postprocessor.visualise_pts(ptspred, 0.05, vis_img)
                vis_combined = np.concatenate((vis_img, curves_mask), axis=1).astype(np.uint8)
                vis_combined = cv2.cvtColor(vis_combined, cv2.COLOR_BGR2RGB)
                val_images.append(wandb.Image(vis_combined, caption=f"Epoch {epoch}: {inputName}"))
    
    metrics = sk_evaluator.summarize_cum(score_threshold=score_threshold, offset_threshold=0.01)
    wandb.log({
        "val_images": val_images,
        "val_f1_cnt": metrics.get('cnt_f1', 0),
        "val_f1_m": metrics.get('m_f1', 0),
    }, step=epoch)
    model.train()
    return metrics


def main():

    config = load_config("config/default.yaml")
    run = wandb.init(project=config["wandb"]["project"], name=config["wandb"]["name"], 
                     dir=config["wandb"]["dir"], config=config)
    # Log default configuration parameters to wandb
    wandb.config.update(config)
    set_seed(config["seed"])
    device = torch.device("cuda")
    #  torch.backends.cudnn.benchmark = True
    
    model, criterion = build_model(config)
    model.to(device)
    train_loader, test_loader = get_dataloaders(config)
    
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
    
    optimizer = torch.optim.AdamW(param_dicts, lr=config.get("lr", 1e-4), weight_decay=config.get("weight_decay", 1e-4))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.get("lr_drop", 20))
    
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if config.get("resume", ""):
        checkpoint = torch.load(config.get("resume"), map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
    
    epochs = config["epochs"]
    
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, criterion, train_loader, optimizer, device, config["clip_max_norm"])
        wandb.log({"train_loss": train_loss}, step=epoch)
        lr_scheduler.step()
        
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'config': config,
        }
        
        if epoch % 20 == 0:
            torch.save(checkpoint, os.path.join(output_dir, f'checkpoint_{epoch}.pth'))
            metrics = validate(model, test_loader, device, epoch, config)
            
    

if __name__ == "__main__":
    main()
