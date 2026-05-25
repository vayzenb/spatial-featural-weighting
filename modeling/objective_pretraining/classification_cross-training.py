# %%
'''Classification cross-training'''

import os
import csv
import json
import io
import shlex
import random
import subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from scipy.optimize import linear_sum_assignment

# %%
# put the main settings in one place
# change encoder_source to switch the frozen encoder condition

config = {
    # choose one: identification, individuation, random
    "encoder_source": "identification",

    # data paths
    "train_images": "/zpool/vladlab/data_drive/stimulus_sets/geogaze_COCO_stim/coco_working/working_v3/train_working3",
    "train_csv": "/zpool/vladlab/data_drive/stimulus_sets/geogaze_COCO_stim/coco_working/working_v3/instances_train_filtered3_bboxes.csv",
    "val_images": "/zpool/vladlab/data_drive/stimulus_sets/geogaze_COCO_stim/coco_working/working_v3/val_working3",
    "val_csv": "/zpool/vladlab/data_drive/stimulus_sets/geogaze_COCO_stim/coco_working/working_v3/instances_val_filtered3_bboxes.csv",
    "test_images": "/zpool/vladlab/data_drive/stimulus_sets/geogaze_COCO_stim/coco_working/working_v3/test_set_official",

    # checkpoints from the two cornet-z training notebooks
    "identification_ckpt": "/zpool/vladlab/data_drive/geogaze_data/cornet_coco_bboxes/cornetz/identification_critical/best.pth.tar",
    "individuation_ckpt": "/zpool/vladlab/data_drive/geogaze_data/cornet_coco_bboxes/cornetz/individuation_critical/best.pth.tar",
    "label_map_path": "/zpool/vladlab/data_drive/geogaze_data/cornet_coco_bboxes/cornetz/identification_critical/label_map.json",

    # output
    "output_root": "/zpool/vladlab/active_drive/omaltz/scripts/geogaze/coco_decoder/cornet_bbox/cornetz/identification/identification_ability/models",

    # model settings
    "num_queries": 10,
    "d_model": 256,
    "nhead": 8,
    "num_decoder_layers": 4,
    "dim_feedforward": 1024,
    "dropout": 0.1,

    # training settings
    "seed": 1,
    "epochs": 200,
    "batch_size": 16,
    "workers": 4,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "ngpus": 1,
    "resume": None,

    # matching costs
    "cost_class": 1.0,
    "cost_bbox": 5.0,
    "cost_giou": 2.0,

    # loss weights
    "no_object_weight": 0.1,
    "loss_bbox": 5.0,
    "loss_giou": 2.0,
}

assert config["encoder_source"] in ["identification", "individuation", "random"]

model_tag = f"{config['encoder_source']}_encoder_identification_head"
outdir = Path(config["output_root"]) / model_tag
outdir.mkdir(parents=True, exist_ok=True)

with open(outdir / "config.json", "w") as f:
    json.dump(config, f, indent=2)

print("encoder source:", config["encoder_source"])
print("output folder:", outdir)

# %%
# set seed and device
# this makes the split/order repeatable

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_gpus(n=1):
    cmd = "nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,nounits"
    gpu_info = subprocess.run(shlex.split(cmd), check=True, stdout=subprocess.PIPE).stdout

    gpus = pd.read_csv(io.BytesIO(gpu_info), sep=", ", engine="python")
    gpus = gpus[gpus["memory.total [MiB]"] > 10000]

    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        visible = [int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
        gpus = gpus[gpus["index"].isin(visible)]

    gpus = gpus.sort_values("memory.free [MiB]", ascending=False)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpus["index"].iloc[:n])

set_seed(config["seed"])

if config["ngpus"] > 0:
    set_gpus(config["ngpus"])

device = torch.device("cuda" if torch.cuda.is_available() and config["ngpus"] > 0 else "cpu")

print("device:", device)
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)

# %%
# point python to the local cornet repo
cornet_repo = Path("/zpool/vladlab/active_drive/omaltz/git_repos/CORnet")
import sys
sys.path.insert(0, str(cornet_repo))

import cornet
print("imported cornet from:", cornet.__file__)

# %%
# load the label map from the identification training run
# this defines the category index order for the identification head

with open(config["label_map_path"], "r") as f:
    label_map = json.load(f)

label_to_idx = label_map["label_to_idx"]
idx_to_label = {int(k): v for k, v in label_map["idx_to_label"].items()}

num_classes = len(label_to_idx)
no_object = num_classes

print("num classes:", num_classes)
print("no-object index:", no_object)

# %%
# set up image transforms
# cornet expects 224 x 224 images with imagenet normalization

normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    normalize,
])

# %%
# dataset for the bbox csv
# this reads in all object boxes and category labels for each image

class DetectionBBoxCSVDataset(torch.utils.data.Dataset):
    def __init__(self, images_root, csv_path, label_to_idx, transform):
        self.images_root = Path(images_root)
        self.csv_path = Path(csv_path)
        self.label_to_idx = label_to_idx
        self.transform = transform
        img_to_anns = defaultdict(list)

        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row["image_file_name"]
                cat_name = row["category_name"].strip()
                if cat_name not in self.label_to_idx:
                    continue

                x = float(row["bbox_x"])
                y = float(row["bbox_y"])
                w = float(row["bbox_w"])
                h = float(row["bbox_h"])

                box = [x, y, x + w, y + h]
                label = self.label_to_idx[cat_name]

                img_to_anns[img_name].append((label, box))

        self.items = []
        missing = 0
        for img_name, anns in img_to_anns.items():
            img_path = self.images_root / img_name

            if img_path.exists():
                self.items.append((img_path, anns))
            else:
                missing += 1
        if len(self.items) == 0:
            raise RuntimeError(f"no matched images found in {self.images_root}")

        if missing:
            print(f"warning: {missing} images from the csv were missing from {self.images_root}")

    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        img_path, ann_list = self.items[idx]
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        labels = []
        boxes = []
        for label, (x1, y1, x2, y2) in ann_list:
            x1 = max(0.0, min(1.0, x1 / orig_w))
            x2 = max(0.0, min(1.0, x2 / orig_w))
            y1 = max(0.0, min(1.0, y1 / orig_h))
            y2 = max(0.0, min(1.0, y2 / orig_h))
            if x2 <= x1 or y2 <= y1:
                continue

            labels.append(label)
            boxes.append([x1, y1, x2, y2])

        labels = torch.tensor(labels, dtype=torch.long)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        image = self.transform(image)

        target = {"boxes_xyxy": boxes, "labels": labels, "image_path": str(img_path)}

        return image, target

# each image can have a different number of boxes
# this keeps targets as a list 
def detection_collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

# %%
# make the train and validation dataloaders

train_ds = DetectionBBoxCSVDataset(config["train_images"], config["train_csv"], label_to_idx, transform)
val_ds = DetectionBBoxCSVDataset(config["val_images"], config["val_csv"], label_to_idx, transform)

train_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["workers"],
    pin_memory=(device.type == "cuda"),
    collate_fn=detection_collate,
)

val_loader = torch.utils.data.DataLoader(
    val_ds,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=config["workers"],
    pin_memory=(device.type == "cuda"),
    collate_fn=detection_collate,
)

print("train images:", len(train_ds))
print("val images:", len(val_ds))

# %%
# box helper functions
# the model predicts boxes as cx, cy, w, h
# the loss uses xyxy in a few places

def cxcywh_to_xyxy(cxcywh):
    cx, cy, w, h = cxcywh.unbind(-1)

    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_area_xyxy(boxes):
    x1, y1, x2, y2 = boxes.unbind(-1)

    return (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
def generalized_box_iou_xyxy(boxes1, boxes2):
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = box_area_xyxy(boxes1)[:, None]
    area2 = box_area_xyxy(boxes2)[None, :]

    union = area1 + area2 - inter
    iou = inter / union.clamp(min=1e-6)

    lt_c = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_c = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])

    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[..., 0] * wh_c[..., 1]
    giou = iou - (area_c - union) / area_c.clamp(min=1e-6)

    return giou

# %%
# positional encoding for the transformer
# this gives each spot in the feature map some information about where it is

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, x):
        b, _, h, w = x.shape
        device = x.device
        y_embed = torch.linspace(0, 1, h, device=device).unsqueeze(1).repeat(1, w)
        x_embed = torch.linspace(0, 1, w, device=device).unsqueeze(0).repeat(h, 1)

        dim_t = torch.arange(self.num_pos_feats, device=device, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2)
        pos = pos.permute(2, 0, 1).unsqueeze(0).repeat(b, 1, 1, 1)

        return pos

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
        elif key.startswith("encoder.model."):
            cornet_state[key.replace("encoder.model.", "")] = value

    if len(cornet_state) == 0:
        raise RuntimeError(f"no cornet-z weights found in {ckpt_path}")

    missing, unexpected = cornet_base.load_state_dict(cornet_state, strict=False)

    print("loaded cornet weights from:", ckpt_path)
    print("missing keys:", len(missing))
    print("unexpected keys:", len(unexpected))

    return cornet_base

# %%
# build the frozen encoder
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
# identification head
# the frozen encoder gives IT features
# the head uses object queries to predict category labels and boxes

class IdentificationHead(nn.Module):
    def __init__(
        self,
        in_ch,
        num_classes,
        num_queries=10,
        d_model=256,
        nhead=8,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.no_object_class = num_classes

        self.input_proj = nn.Conv2d(in_ch, d_model, kernel_size=1)
        self.pos_embed = PositionEmbeddingSine(num_pos_feats=d_model // 2)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.query_embed = nn.Embedding(num_queries, d_model)
        self.class_head = nn.Linear(d_model, num_classes + 1)
        self.box_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 4))

    def forward(self, feats):
        src = self.input_proj(feats)
        pos = self.pos_embed(src)

        b, d, h, w = src.shape

        src_tokens = src.flatten(2).permute(0, 2, 1)
        pos_tokens = pos.flatten(2).permute(0, 2, 1)

        q = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)

        hs = self.decoder(tgt=q, memory=src_tokens + pos_tokens)

        logits = self.class_head(hs)
        boxes = torch.sigmoid(self.box_head(hs))

        return {"pred_logits": logits, "pred_boxes": boxes}

class EncoderWithIdentificationHead(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head
    def forward(self, x):
        feats = self.encoder(x)

        return self.head(feats)

# %%
# build the full model
# first get the IT feature shape, then make the identification head

with torch.no_grad():
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    feats = encoder(dummy)
    in_ch = feats.shape[1]

print("encoder feature shape:", feats.shape)
print("head input channels:", in_ch)

head = IdentificationHead(
    in_ch=in_ch,
    num_classes=num_classes,
    num_queries=config["num_queries"],
    d_model=config["d_model"],
    nhead=config["nhead"],
    num_decoder_layers=config["num_decoder_layers"],
    dim_feedforward=config["dim_feedforward"],
    dropout=config["dropout"],
).to(device)

model = EncoderWithIdentificationHead(encoder, head).to(device)

for p in model.encoder.parameters():
    p.requires_grad = False

for p in model.head.parameters():
    p.requires_grad = True

print(model)

# %%
# match predicted queries to real objects
# this decides which query should be compared to which ground-truth box

def hungarian_matcher(pred_logits, pred_boxes, tgt_labels, tgt_boxes_xyxy, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
    q, _ = pred_logits.shape
    n = tgt_labels.shape[0]

    if n == 0:
        return torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=torch.long)

    prob = pred_logits.softmax(-1)
    cost_cls = -prob[:, tgt_labels]

    pred_xyxy = cxcywh_to_xyxy(pred_boxes).clamp(0, 1)
    cost_l1 = torch.cdist(pred_xyxy, tgt_boxes_xyxy, p=1)

    giou = generalized_box_iou_xyxy(pred_xyxy, tgt_boxes_xyxy)
    cost_giou_matrix = -giou

    cost = cost_class * cost_cls + cost_bbox * cost_l1 + cost_giou * cost_giou_matrix
    cost = cost.detach().cpu().numpy()

    pred_idx, tgt_idx = linear_sum_assignment(cost)

    return torch.as_tensor(pred_idx, dtype=torch.long), torch.as_tensor(tgt_idx, dtype=torch.long)

# %%
# set loss for the identification head
# first match predictions to targets, then calculate class and box losses

class SetCriterion(nn.Module):
    def __init__(
        self,
        num_classes,
        no_object_weight=0.1,
        cost_class=1.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        loss_bbox=5.0,
        loss_giou=2.0,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.no_object = num_classes

        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.loss_bbox = loss_bbox
        self.loss_giou = loss_giou

        weight = torch.ones(num_classes + 1)
        weight[self.no_object] = no_object_weight
        self.register_buffer("ce_weight", weight)

    def forward(self, outputs, targets):
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]

        b, q, _ = pred_logits.shape

        total_ce = 0.0
        total_l1 = 0.0
        total_giou = 0.0
        n_targets = 0

        for i in range(b):
            tgt_labels = targets[i]["labels"]
            tgt_xyxy = targets[i]["boxes_xyxy"]

            n_targets += tgt_labels.shape[0]

            pred_idx, tgt_idx = hungarian_matcher(
                pred_logits[i],
                pred_boxes[i],
                tgt_labels,
                tgt_xyxy,
                cost_class=self.cost_class,
                cost_bbox=self.cost_bbox,
                cost_giou=self.cost_giou,
            )

            target_classes = torch.full((q,), self.no_object, dtype=torch.long, device=pred_logits.device)

            if pred_idx.numel() > 0:
                target_classes[pred_idx.to(pred_logits.device)] = tgt_labels[tgt_idx].to(pred_logits.device)

            ce = F.cross_entropy(pred_logits[i], target_classes, weight=self.ce_weight)
            total_ce += ce

            if pred_idx.numel() > 0:
                p_boxes = pred_boxes[i, pred_idx.to(pred_boxes.device)]
                p_xyxy = cxcywh_to_xyxy(p_boxes).clamp(0, 1)
                t_xyxy = tgt_xyxy[tgt_idx].to(pred_boxes.device)

                l1 = F.l1_loss(p_xyxy, t_xyxy, reduction="mean")
                giou = generalized_box_iou_xyxy(p_xyxy, t_xyxy).diag()
                giou_loss = (1.0 - giou).mean()

                total_l1 += l1
                total_giou += giou_loss

        denom = max(b, 1)

        losses = {
            "loss_ce": total_ce / denom,
            "loss_bbox": total_l1 / denom,
            "loss_giou": total_giou / denom,
        }

        losses["loss_total"] = losses["loss_ce"] + self.loss_bbox * losses["loss_bbox"] + self.loss_giou * losses["loss_giou"]
        losses["n_targets"] = n_targets

        return losses

# %%
# set up loss and optimizer
# only the identification head parameters are updated

criterion = SetCriterion(
    num_classes=num_classes,
    no_object_weight=config["no_object_weight"],
    cost_class=config["cost_class"],
    cost_bbox=config["cost_bbox"],
    cost_giou=config["cost_giou"],
    loss_bbox=config["loss_bbox"],
    loss_giou=config["loss_giou"],
).to(device)

optimizer = torch.optim.AdamW(model.head.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

start_epoch = 0
best_val = float("inf")

print("criterion ready")
print("optimizer ready")

# %%
# run one full pass through either the train or validation set
# train=True updates the head
# train=False just checks the loss

def run_epoch(model, criterion, loader, optimizer, device, train=True, print_every=50):
    model.train() if train else model.eval()
    model.encoder.eval()

    meters = {"loss_total": [], "loss_ce": [], "loss_bbox": [], "loss_giou": []}

    for it, (images, targets) in enumerate(loader):
        images = torch.stack(images, dim=0).to(device)

        target_list = []
        for t in targets:
            target_list.append({
                "boxes_xyxy": t["boxes_xyxy"].to(device),
                "labels": t["labels"].to(device),
                "image_path": t.get("image_path", ""),
            })

        with torch.set_grad_enabled(train):
            outputs = model(images)
            losses = criterion(outputs, target_list)

            if train:
                optimizer.zero_grad(set_to_none=True)
                losses["loss_total"].backward()
                optimizer.step()

        for key in meters:
            meters[key].append(float(losses[key].detach().cpu()))

        if (it + 1) % print_every == 0:
            split = "train" if train else "val"

            print(
                f"{split} iter {it + 1}/{len(loader)} "
                f"loss={np.mean(meters['loss_total'][-print_every:]):.4f} "
                f"ce={np.mean(meters['loss_ce'][-print_every:]):.4f} "
                f"bbox={np.mean(meters['loss_bbox'][-print_every:]):.4f} "
                f"giou={np.mean(meters['loss_giou'][-print_every:]):.4f}"
            )

    return {key: float(np.mean(values)) for key, values in meters.items()}

# %%
# train the identification head
# the cornet-z IT encoder stays frozen the whole time

history = []

for epoch in range(start_epoch, config["epochs"]):
    print("\n" + "=" * 100)
    print(f"epoch {epoch}/{config['epochs'] - 1}")

    train_stats = run_epoch(model, criterion, train_loader, optimizer, device, train=True, print_every=50)
    val_stats = run_epoch(model, criterion, val_loader, optimizer, device, train=False, print_every=50)

    print("train:", train_stats)
    print("val:  ", val_stats)

    is_best = val_stats["loss_total"] < best_val
    if is_best:
        best_val = val_stats["loss_total"]

    ckpt = {
        "epoch": epoch,
        "best_val": best_val,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "train_stats": train_stats,
        "val_stats": val_stats,
        "label_map": label_map,
    }

    torch.save(ckpt, outdir / "latest.pth.tar")
    torch.save(ckpt, outdir / f"epoch_{epoch:03d}.pth.tar")

    if is_best:
        torch.save(ckpt, outdir / "best.pth.tar")
        print(f"saved best checkpoint: {outdir / 'best.pth.tar'}")

    history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()}, **{f"val_{k}": v for k, v in val_stats.items()}})
    pd.DataFrame(history).to_csv(outdir / "training_history.csv", index=False)

print("done training")

# %%
# load the best checkpoint before running the test set

best_ckpt = torch.load(outdir / "best.pth.tar", map_location=device)
model.load_state_dict(best_ckpt["model"], strict=False)
model.eval()
model.encoder.eval()

print("loaded best checkpoint from:", outdir / "best.pth.tar")
print("best val:", best_ckpt["best_val"])

# %%
# run the test set and save the image/category csv
# each row is one detected category for one image
# if the model predicts no objects, the image gets a no_object row

valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
image_folder = Path(config["test_images"])
out_csv = outdir / f"{model_tag}.csv"

image_paths = sorted([p for p in image_folder.iterdir() if p.suffix.lower() in valid_exts])

print("found test images:", len(image_paths))
print("output csv:", out_csv)

rows = []

model.eval()

with torch.no_grad():
    for i, image_path in enumerate(image_paths, start=1):
        image = Image.open(image_path).convert("RGB")
        x = transform(image).unsqueeze(0).to(device)
        outputs = model(x)
        pred_logits = outputs["pred_logits"][0]
        probs = F.softmax(pred_logits, dim=-1)
        scores, labels = probs.max(dim=-1)
        found_labels = set()
        for label in labels:
            label_idx = label.item()
            if label_idx == no_object:
                continue
            found_labels.add(idx_to_label[label_idx])
        if len(found_labels) == 0:
            rows.append({"image_file_name": image_path.name, "category_name": "no_object"})
        else:
            for label_name in sorted(found_labels):
                rows.append({"image_file_name": image_path.name, "category_name": label_name})

        if i % 50 == 0 or i == len(image_paths):
            print(f"processed {i}/{len(image_paths)} images")
pred_df = pd.DataFrame(rows)
pred_df.to_csv(out_csv, index=False)

print("saved csv to:", out_csv)
print("number of rows:", len(pred_df))
pred_df.head(20)


