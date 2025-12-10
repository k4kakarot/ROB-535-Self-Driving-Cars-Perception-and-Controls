import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from model_vla import (
    ACTIONS,
    ACTION_TO_IDX,
    IMG_TRANSFORM,
    parse_command_to_action,
    action_to_one_hot,
    VisionEncoder,
    VLAPolicy,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ====== CoVLA dataset wrapper ======

class CoVLATorchDataset(Dataset):
    

    def __init__(self, hf_dataset, speed_threshold: float = 10.0):
        super().__init__()
        self.ds = hf_dataset

        # Precompute indices that have some caption text
        self.indices = []
        for i in range(len(self.ds)):
            s = self.ds[i]
            if any(k in s for k in ["rich_caption", "plain_caption", "caption", "text"]):
                self.indices.append(i)

        print(f"Using {len(self.indices)} caption samples out of {len(self.ds)} total rows.")

    def __len__(self):
        return len(self.indices)

    def _scene_to_action(self, caption: str, risk_text: str, has_pedestrian: bool) -> str:
        """
        Heuristic label generator.

        Priority:
        1) TURN_LEFT   – if left-turn phrases in caption or risk text
        2) TURN_RIGHT  – if right-turn phrases in caption or risk text
        3) SLOW_DOWN   – pedestrians / crosswalk / heavy traffic / careful
        4) MAINTAIN_SPEED – otherwise
        """
        txt = (caption or "") + " " + (risk_text or "")
        txt = txt.lower()

        # Left turn
        if any(kw in txt for kw in ["left turn", "turn left", "veers left", "left-hand turn"]):
            return "TURN_LEFT"

        # Right turn
        if any(kw in txt for kw in ["right turn", "turn right", "veers right", "right-hand turn"]):
            return "TURN_RIGHT"

        # Slow down for pedestrians / hazards
        if has_pedestrian:
            return "SLOW_DOWN"

        if any(kw in txt for kw in [
            "pedestrian",
            "pedestrians",
            "crosswalk",
            "intersection",
            "merge",
            "heavy traffic",
            "congestion",
            "careful",
            "distance",
            "brake",
        ]):
            return "SLOW_DOWN"

        return "MAINTAIN_SPEED"


    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        sample = self.ds[real_idx]

        # 1) Image: ensure RGB
        image = sample["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_tensor = IMG_TRANSFORM(image)

        # 2) Caption: try multiple keys (we know at least one exists by construction)
        caption = None
        for key in ["rich_caption", "plain_caption", "caption", "text"]:
            if key in sample and sample[key] is not None:
                caption = sample[key]
                break
        if caption is None:
            caption = ""

        # 3) Risk & has_pedestrian for the label
        risk_text = sample.get("risk", "")
        has_pedestrian = bool(sample.get("has_pedestrian", False))

        action_label = self._scene_to_action(caption, risk_text, has_pedestrian)

        if action_label not in ACTIONS:
            raise ValueError(f"Unexpected action_label '{action_label}'")

        target_idx = ACTION_TO_IDX[action_label]

        # 4) Language feature: parse caption → coarse action → one-hot
        parsed_cmd_action = parse_command_to_action(caption)
        cmd_vec = action_to_one_hot(parsed_cmd_action)

        return image_tensor, cmd_vec, target_idx





# ====== Training utilities ======

def train_one_epoch(model, loader, optimizer, epoch: int):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, cmd_vecs, targets in loader:
        images = images.to(DEVICE)
        cmd_vecs = cmd_vecs.to(DEVICE)
        targets = targets.to(DEVICE)

        logits = model(images, cmd_vecs)
        loss = F.cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    print(f"Epoch {epoch}: train loss={avg_loss:.4f}, acc={avg_acc:.4f}")
    return avg_loss, avg_acc


@torch.no_grad()
def eval_model(model, loader, split_name: str = "val"):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, cmd_vecs, targets in loader:
        images = images.to(DEVICE)
        cmd_vecs = cmd_vecs.to(DEVICE)
        targets = targets.to(DEVICE)

        logits = model(images, cmd_vecs)
        loss = F.cross_entropy(logits, targets)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    print(f"{split_name}: loss={avg_loss:.4f}, acc={avg_acc:.4f}")
    return avg_loss, avg_acc


# ====== Main training entry point ======

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--speed_threshold", type=float, default=10.0)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ✅ Use the *public* CoVLA-Mini mirror (no gating):
    hf_full = load_dataset("the-future-dev/CoVLA-Dataset-Mini", split="train")

    # Train/val split (e.g., 80/20)
    split = hf_full.train_test_split(test_size=0.2, seed=42)
    hf_train = split["train"]
    hf_val = split["test"]

    train_ds = CoVLATorchDataset(hf_train, speed_threshold=args.speed_threshold)
    val_ds = CoVLATorchDataset(hf_val, speed_threshold=args.speed_threshold)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model
    vision_encoder = VisionEncoder(pretrained=True, freeze=True).to(DEVICE)
    model = VLAPolicy(vision_encoder, num_actions=len(ACTIONS)).to(DEVICE)

    # Only train the MLP head
    optimizer = torch.optim.Adam(model.mlp.parameters(), lr=args.lr)

    best_val_acc = 0.0
    best_ckpt_path = os.path.join(args.checkpoint_dir, "vla_covla_best.pt")

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, epoch)
        _, val_acc = eval_model(model, val_loader, split_name="val")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"==> Saved new best model (val acc={val_acc:.4f})")

    print(f"Training done. Best val acc={best_val_acc:.4f}")
    print(f"Best checkpoint saved to: {best_ckpt_path}")


if __name__ == "__main__":
    main()
