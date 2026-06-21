import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    get_scheduler,
)
from torch.optim import AdamW
from tqdm import tqdm

# =============================================================================
# CONFIG
# =============================================================================

DATASET_DIR = r"C:\Users\srush\Desktop\sim_dataset"
IMAGE_DIR   = os.path.join(DATASET_DIR, "images")
LABEL_DIR   = os.path.join(DATASET_DIR, "labels")
OUTPUT_DIR  = r"C:\Users\srush\Desktop\yolos_finetuned"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Must match exact order in your classes.txt
CLASSES  = ["robot_arm", "cube_red", "cube_green", "conveyor", "bin_red", "bin_green"]
ID2LABEL = {i: c for i, c in enumerate(CLASSES)}
LABEL2ID = {c: i for i, c in enumerate(CLASSES)}

NUM_EPOCHS = 30
BATCH_SIZE = 2
LR         = 2e-5
IMG_SIZE   = 640

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")


# =============================================================================
# DATASET — returns raw image + target, processor called in collate_fn
# =============================================================================

class SimDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.samples   = sorted([
            f for f in os.listdir(image_dir) if f.endswith(".jpg")
        ])
        print(f"[INFO] Found {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name   = self.samples[idx]
        label_name = img_name.replace(".jpg", ".txt")

        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        W, H  = image.size

        boxes  = []
        labels = []

        label_path = os.path.join(self.label_dir, label_name)
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, xc, yc, bw, bh = map(float, parts)

                    # YOLO normalised → absolute xyxy
                    x_min = (xc - bw / 2) * W
                    y_min = (yc - bh / 2) * H
                    x_max = (xc + bw / 2) * W
                    y_max = (yc + bh / 2) * H

                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(int(cls))

        if not boxes:
            boxes  = [[0.0, 0.0, 1.0, 1.0]]
            labels = [0]

        target = {
            "image_id": idx,
            "annotations": [
                {
                    "id": j,
                    "image_id": idx,
                    "category_id": labels[j],
                    "bbox": [
                        boxes[j][0],
                        boxes[j][1],
                        boxes[j][2] - boxes[j][0],
                        boxes[j][3] - boxes[j][1],
                    ],
                    "area": (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1]),
                    "iscrowd": 0,
                }
                for j in range(len(boxes))
            ],
        }

        return image, target


# =============================================================================
# COLLATE — processor runs here on a batch of PIL images
# =============================================================================

def make_collate_fn(processor):
    def collate_fn(batch):
        images  = [item[0] for item in batch]
        targets = [item[1] for item in batch]

        encoding = processor(
            images=images,
            annotations=targets,
            return_tensors="pt",
        )

        # encoding["labels"] is a list of dicts with keys: class_labels, boxes
        return {
            "pixel_values": encoding["pixel_values"],
            "labels":       encoding["labels"],
        }
    return collate_fn


# =============================================================================
# TRAINING
# =============================================================================

def train():
    print("[INFO] Loading hustvl/yolos-tiny ...")
    processor = AutoImageProcessor.from_pretrained(
        "hustvl/yolos-tiny",
        size={"max_height": IMG_SIZE, "max_width": IMG_SIZE},
    )
    model = AutoModelForObjectDetection.from_pretrained(
        "hustvl/yolos-tiny",
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    model.to(DEVICE)

    dataset    = SimDataset(IMAGE_DIR, LABEL_DIR)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=make_collate_fn(processor),
    )

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=5,
        num_training_steps=NUM_EPOCHS * len(dataloader),
    )

    print(f"[INFO] Training for {NUM_EPOCHS} epochs on {len(dataset)} images ...\n")

    best_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1:02d}/{NUM_EPOCHS}"):
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = [
                {k: v.to(DEVICE) for k, v in lbl.items()}
                for lbl in batch["labels"]
            ]

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss    = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"  Epoch {epoch+1:02d} — loss: {avg_loss:.4f}")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(OUTPUT_DIR, "best")
            model.save_pretrained(best_path)
            processor.save_pretrained(best_path)
            print(f"  [BEST] loss={best_loss:.4f} -> {best_path}")

        # Checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            ckpt = os.path.join(OUTPUT_DIR, f"checkpoint_epoch{epoch+1}")
            model.save_pretrained(ckpt)
            processor.save_pretrained(ckpt)
            print(f"  [CKPT] -> {ckpt}")

    # Final save
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"\n[DONE] Final model  -> {OUTPUT_DIR}")
    print(f"[DONE] Best model   -> {OUTPUT_DIR}\\best")
    print(f"[DONE] Best loss    -> {best_loss:.4f}")


if __name__ == "__main__":
    train()