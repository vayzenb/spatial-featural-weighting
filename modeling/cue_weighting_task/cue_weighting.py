'''
Train then test models on cue weighting task
'''

# %% [markdown]
# ## Run cue weighting task (training and testing)
# 

# %%
import os
import re
import random
import json
import io
import shlex
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T

# %%
# put the main settings in one place
# encoder_source is the only thing you change to switch model condition

config = {
    # choose one: identification, individuation, random
    "encoder_source": "individuation",

    # data paths
    "pairs_dir": "/zpool/vladlab/data_drive/geogaze_data/decoder_stimulus/pairs",
    "left_masks_dir": "/zpool/vladlab/data_drive/geogaze_data/decoder_stimulus/left_masks",
    "right_masks_dir": "/zpool/vladlab/data_drive/geogaze_data/decoder_stimulus/right_masks",

    # checkpoints from the two cornet-z training notebooks
    "identification_ckpt": "/zpool/vladlab/data_drive/geogaze_data/cornet_coco_bboxes/cornetz/identification_critical/best.pth.tar",
    "individuation_ckpt": "/zpool/vladlab/data_drive/geogaze_data/cornet_coco_bboxes/cornetz/individuation_critical/best.pth.tar",

    # output root for this geogaze task
    "output_root": "/zpool/vladlab/data_drive/geogaze_data/geogaze_final/cornetz_geogaze_task",

    # test image folders
    # old identification and individuation notebooks used the identification test_images folder
    # old random notebook used the randomized test_images folder
    "test_image_roots": {
        "identification": "/zpool/vladlab/data_drive/geogaze_data/decoder_stimulus/test_stimuli/identification/test_images",
        "individuation": "/zpool/vladlab/data_drive/geogaze_data/decoder_stimulus/test_stimuli/identification/test_images",
        "random": "/zpool/vladlab/data_drive/geogaze_data/decoder_stimulus/test_stimuli/randomized/test_images",
    },

    # run all 16 mask-side / pair conditions
    "run_conditions": [
        ("L", "bc_bs"),
        ("L", "bc_gc"),
        ("L", "bs_bc"),
        ("L", "bs_gs"),
        ("L", "gc_bc"),
        ("L", "gc_gs"),
        ("L", "gs_bs"),
        ("L", "gs_gc"),
        ("R", "bc_bs"),
        ("R", "bc_gc"),
        ("R", "bs_bc"),
        ("R", "bs_gs"),
        ("R", "gc_bc"),
        ("R", "gc_gs"),
        ("R", "gs_bs"),
        ("R", "gs_gc"),
    ],

    # training settings copied from the individuation geogaze notebook
    "seed": 1,
    "epochs": 200,
    "batch_size": 32,
    "workers": 8,
    "val_split": 0.2,
    "lr": 0.01,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "threshold": 0.2,

    # image settings
    "img_size": 224,
}

assert config["encoder_source"] in ["identification", "individuation", "random"]

pairs_dir = Path(config["pairs_dir"])
output_root = Path(config["output_root"]) / config["encoder_source"]
test_img_root = Path(config["test_image_roots"][config["encoder_source"]])

output_root.mkdir(parents=True, exist_ok=True)

print("encoder source:", config["encoder_source"])
print("output root:", output_root)
print("test image root:", test_img_root)
print("number of geogaze runs:", len(config["run_conditions"]))

# %%
# set seed and device

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(config["seed"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device:", device)
print("torch:", torch.__version__)

# %%
# image transforms
# cornet expects 224 x 224 images with imagenet normalization

img_tf = T.Compose([
    T.Resize((config["img_size"], config["img_size"])),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_mask_binary(path):
    # object pixels are black in the masks
    mask = Image.open(path).convert("L")
    mask = mask.resize((config["img_size"], config["img_size"]), resample=Image.NEAREST)
    mask = np.asarray(mask, dtype=np.uint8)
    mask = (mask == 0).astype(np.float32)

    return torch.from_numpy(mask).unsqueeze(0)

# %%
# dataset for image-mask pairs
# each item returns the pair image, the binary mask, and an id string

class PairMaskDataset(Dataset):
    def __init__(self, items, img_transform):
        self.items = items
        self.img_transform = img_transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path, mid, id_ = self.items[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.img_transform(image)

        mask = load_mask_binary(mask_path)

        return image, mask, f"{mid}_{id_}"

# %%
# collect image-mask pairs for one mask side and one pair condition

def get_masks_dir(mask_side):
    if mask_side == "L":
        return Path(config["left_masks_dir"])

    if mask_side == "R":
        return Path(config["right_masks_dir"])

    raise ValueError("mask_side must be L or R")

def collect_items(mask_side, pair_mid):
    masks_dir = get_masks_dir(mask_side)
    pair_re = re.compile(rf"^pair_{re.escape(pair_mid)}_(\d+)\.png$")

    items = []
    skipped = 0

    for fn in os.listdir(pairs_dir):
        match = pair_re.match(fn)
        if not match:
            continue

        id_ = match.group(1)
        img_path = pairs_dir / fn
        mask_path = masks_dir / f"mask{mask_side}_{pair_mid}_{id_}.png"

        if not mask_path.is_file():
            skipped += 1
            continue

        items.append((img_path, mask_path, pair_mid, id_))

    items.sort(key=lambda x: int(x[3]))

    if skipped:
        print(f"warning: skipped {skipped} image(s) with no matching mask{mask_side}")

    return items

# %%
# make train and validation loaders for one condition

def make_loaders(items):
    rng = random.Random(config["seed"])
    shuffled = items[:]
    rng.shuffle(shuffled)

    n_val = max(1, int(len(shuffled) * config["val_split"]))
    val_items = shuffled[:n_val]
    train_items = shuffled[n_val:]

    train_loader = DataLoader(
        PairMaskDataset(train_items, img_tf),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["workers"],
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        PairMaskDataset(val_items, img_tf),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["workers"],
        pin_memory=(device.type == "cuda"),
    )

    return train_loader, val_loader, train_items, val_items

# %%
# point python to the local cornet repo
# this lets us build cornet-z the same way for all encoder sources

cornet_repo = Path("/zpool/vladlab/active_drive/omaltz/git_repos/CORnet")

import sys
sys.path.insert(0, str(cornet_repo))

import cornet

print("imported cornet from:", cornet.__file__)

# %%
# cornet-z IT feature extractor
# this runs the full cornet model, but saves the IT layer output with a hook

class CornetITSpatialBackbone(nn.Module):
    def __init__(self, cornet_model):
        super().__init__()

        self.model = cornet_model
        self.feat = None

        if not hasattr(self.model, "IT"):
            raise ValueError("cornet model does not have an IT layer")

        def hook_fn(module, inp, out):
            self.feat = out

        self.model.IT.register_forward_hook(hook_fn)

    def forward(self, x):
        _ = self.model(x)

        if self.feat is None:
            raise RuntimeError("IT hook did not capture an output")

        return self.feat

# %%
# load cornet-z weights from a trained checkpoint
# this ignores the old identification/individuation heads

def load_cornet_weights_from_checkpoint(cornet_base, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt

    cornet_state = {}

    for key, value in state.items():
        if key.startswith("backbone.model."):
            cornet_state[key.replace("backbone.model.", "")] = value
        elif key.startswith("module.backbone.model."):
            cornet_state[key.replace("module.backbone.model.", "")] = value

    if len(cornet_state) == 0:
        raise RuntimeError(f"no cornet-z weights found in {ckpt_path}")

    missing, unexpected = cornet_base.load_state_dict(cornet_state, strict=False)

    print("loaded cornet weights from:", ckpt_path)
    print("missing keys:", len(missing))
    print("unexpected keys:", len(unexpected))

    return cornet_base

# %%
# build the frozen encoder once
# identification and individuation load trained cornet-z weights
# random keeps the cornet-z weights random

cornet_base = cornet.cornet_z(pretrained=False, map_location=device)

if hasattr(cornet_base, "module"):
    cornet_base = cornet_base.module

if config["encoder_source"] == "identification":
    cornet_base = load_cornet_weights_from_checkpoint(cornet_base, config["identification_ckpt"])
elif config["encoder_source"] == "individuation":
    cornet_base = load_cornet_weights_from_checkpoint(cornet_base, config["individuation_ckpt"])
else:
    print("using random cornet-z weights")

encoder = CornetITSpatialBackbone(cornet_base).to(device)

for p in encoder.parameters():
    p.requires_grad = False

encoder.eval()

print("has IT layer:", hasattr(cornet_base, "IT"))
print("encoder source:", config["encoder_source"])

# %%
# mask head
# takes the IT feature map and predicts a one-channel object mask

class MaskHead(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, 1, kernel_size=1)
    def forward(self, feats):
        logits = self.proj(feats)
        logits = F.interpolate(
            logits,
            size=(config["img_size"], config["img_size"]),
            mode="bilinear",
            align_corners=False,
        )
        return logits

class CORnetMaskModel(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x):
        feats = self.encoder(x)
        logits = self.head(feats)
        return logits

# %%
# find how many IT channels the mask head needs

with torch.no_grad():
    dummy = torch.zeros(1, 3, config["img_size"], config["img_size"], device=device)
    feats = encoder(dummy)
    in_ch = feats.shape[1]

print("encoder feature shape:", feats.shape)
print("mask head input channels:", in_ch)

# %%
# build new mask model for one condition
# each condition gets a new head, but the frozen encoder source stays the same

def build_mask_model():
    head = MaskHead(in_ch).to(device)
    model = CORnetMaskModel(encoder, head).to(device)

    for p in model.encoder.parameters():
        p.requires_grad = False

    for p in model.head.parameters():
        p.requires_grad = True

    return model

# %%
# train one geogaze head

def train_one_condition(model, train_loader, val_loader, run_dir):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(
        model.head.parameters(),
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )
    best_val_loss = float("inf")
    history = []

    for epoch in range(config["epochs"]):
        model.train()
        model.encoder.eval()
        train_loss = 0.0
        for images, masks, ids in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks, ids in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                loss = criterion(logits, masks)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.head.state_dict(), run_dir / "mask_head_best.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                },
                run_dir / "best_full_model.pt",
            )

        torch.save(model.head.state_dict(), run_dir / "mask_head_latest.pt")
        print(
            f"epoch {epoch + 1}/{config['epochs']} | "
            f"train loss: {train_loss:.4f} | "
            f"val loss: {val_loss:.4f}"
        )
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)

    return best_val_loss

# %%
# save probability maps for the test images
# find test images in test_image_root / model_tag
# then save each probability map csv inside the run_dir

def save_test_probability_maps(model, model_tag, run_dir):
    pred_map_folder = test_img_root / model_tag
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

    if not pred_map_folder.exists():
        print(f"warning: no test image folder found: {pred_map_folder}")
        return 0

    image_paths = sorted([p for p in pred_map_folder.iterdir() if p.is_file() and p.suffix.lower() in exts])

    print(f"found {len(image_paths)} test images in: {pred_map_folder}")

    model.eval()

    with torch.no_grad():
        for img_path in image_paths:
            image = Image.open(img_path).convert("RGB")
            x = img_tf(image).unsqueeze(0).to(device)

            logits = model(x)
            probs = torch.sigmoid(logits)

            prob_map = probs.squeeze().detach().cpu().numpy()

            out_name = f"{img_path.stem}_prob_map.csv"
            out_path = run_dir / out_name

            pd.DataFrame(prob_map).to_csv(out_path, index=False)

            print(f"saved: {out_path}")

    return len(image_paths)

# %%
# run all mask-side / pair conditions
# each condition gets its own fresh head, output folder, checkpoints, history csv, and test probability csvs

summary_rows = []

for run_idx, (mask_side, pair_mid) in enumerate(config["run_conditions"], start=1):
    model_tag = f"cornetz_{config['encoder_source']}_mask{mask_side}_{pair_mid}"
    run_dir = output_root / model_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    run_config = dict(config)
    run_config["mask_side"] = mask_side
    run_config["pair_mid"] = pair_mid
    run_config["model_tag"] = model_tag
    run_config["run_dir"] = str(run_dir)

    with open(run_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    print("\n" + "=" * 100)
    print(f"run {run_idx}/{len(config['run_conditions'])}: {model_tag}")

    items = collect_items(mask_side, pair_mid)

    if len(items) == 0:
        print(f"warning: no training items found for {model_tag}; skipping")
        summary_rows.append({
            "model_tag": model_tag,
            "mask_side": mask_side,
            "pair_mid": pair_mid,
            "n_train_items": 0,
            "n_val_items": 0,
            "best_val_loss": np.nan,
            "n_test_csvs": 0,
            "status": "skipped_no_training_items",
        })
        continue

    train_loader, val_loader, train_items, val_items = make_loaders(items)

    print("train:", len(train_items))
    print("val:", len(val_items))

    model = build_mask_model()
    best_val_loss = train_one_condition(model, train_loader, val_loader, run_dir)

    # reload the best head before making the test csvs
    best_head_path = run_dir / "mask_head_best.pt"
    model.head.load_state_dict(torch.load(best_head_path, map_location=device))

    n_test_csvs = save_test_probability_maps(model, model_tag, run_dir)

    summary_rows.append({
        "model_tag": model_tag,
        "mask_side": mask_side,
        "pair_mid": pair_mid,
        "n_train_items": len(train_items),
        "n_val_items": len(val_items),
        "best_val_loss": best_val_loss,
        "n_test_csvs": n_test_csvs,
        "status": "done",
    })

summary = pd.DataFrame(summary_rows)
summary_path = output_root / f"summary_{config['encoder_source']}.csv"
summary.to_csv(summary_path, index=False)

print("\nall runs done")
print("summary saved to:", summary_path)
summary


