import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForObjectDetection


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList([nn.Linear(d, dims[i+1]) for i, d in enumerate(dims[:-1])])
        self.num_layers = num_layers

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i < self.num_layers - 1:
                x = F.relu(layer(x))
            else:
                x = layer(x)
        return x


class BlumNet(nn.Module):
    """
    BlumNet model that uses the Huggingface Deformable DETR model with custom modifications
    to mimic the behavior of the custom model in deformable_detr.py.
    This version excludes gid and auxiliary loss implementations.
    """
    def __init__(self, load_path, points_per_path, out_pts, num_queries, device):
        super().__init__()
        self.load_path = load_path
        self.points_per_path = points_per_path
        self.out_pts = out_pts
        self.num_queries = num_queries
        self.device = device

        # Load pre-trained Huggingface Deformable DETR
        self.processor = AutoImageProcessor.from_pretrained(
            "SenseTime/deformable-detr",
            use_fast=True,
        )
        self.deformable_detr = AutoModelForObjectDetection.from_pretrained(
            "SenseTime/deformable-detr",
            ignore_mismatched_sizes=True,
            num_labels=2,
            num_queries=self.num_queries + self.out_pts,
            num_feature_levels=3,
        )

        # --- Custom Query Embedding ---
        hidden_dim = self.deformable_detr.config.d_model

        # --- Iterative Bounding Box Head for Curves ---
        num_decoder_layers = self.deformable_detr.config.decoder_layers
        self.bbox_embed = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 2 * points_per_path, 3)
            for _ in range(num_decoder_layers)
        ])
        
        # --- Classification Heads ---
        num_classes = 2  # [target, non-target]
        self.class_embed = nn.Linear(hidden_dim, num_classes)       # for curve classification
        self.class_pt_embed = nn.Linear(hidden_dim, num_classes+1)    # for point classification

        # --- Point Regression Head ---
        self.pt_embed = MLP(hidden_dim, hidden_dim, 2, 3)         

        if load_path != "pretrain":
            print(f"Loading model from {load_path}")
            self.load_state_dict(torch.load(load_path))
        else:
            print("Loading pretrained model from Huggingface SenseTime/deformable-detr")



    def forward(self, images):
        """
        Forward pass:
          1. Prepare input and call the Huggingface Deformable DETR backbone/transformer.
          2. Apply iterative refinement for curve predictions.
          3. Process point predictions (without iterative refinement).
          4. Return only the final predictions (no auxiliary losses).
        """
        # --- Input Handling ---
        if isinstance(images, dict):
            tensor = images["tensors"].to(self.device)
            mask = images["mask"].to(self.device)
        else:
            tensor = images.tensors.to(self.device)
            mask = images.mask.to(self.device)
            
        inputs = self.processor(images=list(tensor), return_tensors="pt")
        outputs = self.deformable_detr(
            pixel_values=inputs["pixel_values"].to(self.device),
            pixel_mask=mask.to(self.device),
            output_hidden_states=True,
            return_dict=True,
        )

        # --- Access Decoder Hidden States ---
        hs = outputs.decoder_hidden_states
        batch_size = tensor.shape[0]
        
        # Check if decoder hidden states are available
        if hs is not None and len(hs) > 0:
            num_layers = len(hs)
            
            # Initialize reference points from first layer
            first_hs = hs[0]
            curve_hs_0 = first_hs[:, self.out_pts:]
            reference_points = self.bbox_embed[0](curve_hs_0).sigmoid()
            
            # Iteratively refine reference points if we have multiple decoder layers
            if num_layers > 1:
                # Use the minimum between available decoder layers and bbox_embed layers
                refinement_layers = min(num_layers, len(self.bbox_embed))
                for i in range(1, refinement_layers):
                    layer_hs = hs[i]
                    curve_hs_i = layer_hs[:, self.out_pts:]
                    
                    # Apply the current bbox_embed to get delta coordinates
                    delta = self.bbox_embed[i](curve_hs_i)
                    
                    # Convert reference points for refinement
                    inv_sigmoid_ref = inverse_sigmoid(reference_points)
                    
                    # Add delta to reference points
                    if delta.shape[-1] == inv_sigmoid_ref.shape[-1]:
                        refined = delta + inv_sigmoid_ref
                    else:
                        # Handle mismatch in dimensions by repeating reference points
                        repeat_num = delta.shape[-1] // inv_sigmoid_ref.shape[-1]
                        refined = delta + torch.cat([inv_sigmoid_ref for _ in range(repeat_num)], dim=-1)
                    
                    # Update reference points with refined coordinates
                    reference_points = refined.sigmoid()
            
            # Get the final hidden states for classification
            last_hs = hs[-1]
            
            # Classification for curves
            curve_hs = last_hs[:, self.out_pts:]
            curve_logits = self.class_embed(curve_hs)
            curve_boxes = reference_points  # Use the refined boxes
            
            # Classification for points
            if self.out_pts > 0:
                point_hs = last_hs[:, :self.out_pts]
                point_logits = self.class_pt_embed(point_hs)
                point_boxes = self.pt_embed(point_hs).sigmoid()
            else:
                point_logits = torch.zeros((batch_size, 0, 2), device=self.device)
                point_boxes = torch.zeros((batch_size, 0, 2), device=self.device)
        else:
            # Fallback if decoder hidden states aren't available
            print("Warning: No decoder hidden states returned by the model")
            curve_logits = torch.zeros((batch_size, self.num_queries, 2), device=self.device)
            curve_boxes = torch.zeros((batch_size, self.num_queries, 2 * self.points_per_path), device=self.device)
            point_logits = torch.zeros((batch_size, self.out_pts, 2), device=self.device)
            point_boxes = torch.zeros((batch_size, self.out_pts, 2), device=self.device)

        out = {
            "curves": {
                "pred_logits": curve_logits,
                "pred_boxes": curve_boxes,
            },
            "pts": {
                "pred_logits": point_logits,
                "pred_boxes": point_boxes,
            }
        }
        return out


def build_model(config, device):
    # Build the model using the provided configuration and device
    model = BlumNet(
        config["load_path"],
        config["points_per_path"],
        config["out_pts"],
        config["num_queries"],
        device
    )
    model.to(device)
    return model


if __name__ == '__main__':
    import argparse
    import yaml
    from types import SimpleNamespace
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    args = parser.parse_args()

    # Use default configs if the file doesn't exist
    try:
        with open(args.config, "r") as f:
            configs = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file {args.config} not found, using default values")
        configs = {
            "load_path": "pretrain",
            "points_per_path": 10,
            "out_pts": 32,
            "num_queries": 100
        }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(configs, device)
    model.eval()

    # Create dummy inputs as a batch of 3 images
    batch_size = 3
    height, width = 480, 640
    dummy_inputs = {
        "tensors": torch.rand(batch_size, 3, height, width),
        "mask": torch.zeros(batch_size, height, width, dtype=torch.bool)
    }
    with torch.no_grad():
        output = model(dummy_inputs)
        print("Output keys:", output.keys())
        print("Curves logits shape:", output["curves"]["pred_logits"].shape)
        print("Curves boxes shape:", output["curves"]["pred_boxes"].shape)
        print("Points logits shape:", output["pts"]["pred_logits"].shape)
        print("Points boxes shape:", output["pts"]["pred_boxes"].shape)
