# %%
'''Individuation cross-training'''

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
# change encoder_source to switch the frozen encoder condition

config = {
    # identification, individuation, random
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

    # output
    "model_output_root": "/zpool/vladlab/active_drive/omaltz/scripts/geogaze/coco_decoder/cornet_bbox/cornetz/individuation/individuation_ability/models",
    "results_output_root": "/zpool/vladlab/active_drive/omaltz/scripts/geogaze/coco_decoder/cornet_bbox/cornetz/individuation/individuation_ability/results",

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
    "cost_obj": 1.0,
    "cost_bbox": 5.0,
    "cost_giou": 2.0,

    # loss weights
    "no_object_weight": 0.1,
    "loss_bbox": 5.0,
    "loss_giou": 2.0,

    # test settings
    "conf_threshold": 0.7,
    "top_k": 6,
}

assert config["encoder_source"] in ["identification", "individuation", "random"]

#  the random condition
tag_name = "randomized" if config["encoder_source"] == "random" else config["encoder_source"]
model_tag = f"{tag_name}_encoder_individuation_head"

outdir = Path(config["model_output_root"]) / model_tag
outdir.mkdir(parents=True, exist_ok=True)

results_dir = Path(config["results_output_root"])
results_dir.mkdir(parents=True, exist_ok=True)

out_csv = results_dir / f"{model_tag}.csv"

# save the config so we know exactly what was run
with open(outdir / "config.json", "w") as f:
    json.dump(config, f, indent=2)

print("encoder source:", config["encoder_source"])
print("model tag:", model_tag)
print("output folder:", outdir)
print("test csv:", out_csv)

# %%
# set seed and device
# this makes the run more repeatable

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
# pick the gpu with the most free memory
# this matches the style we used in the training notebooks

def set_gpus(n=1):
    cmd = "nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,nounits"
    gpu_info = subprocess.run(shlex.split(cmd), check=True, stdout=subprocess.PIPE).stdout

    gpus = pd.read_csv(io.BytesIO(gpu_info), sep=", ", engine="python")
    gpus = gpus[gpus["memory.total [MiB]"] > 10000]
    # if cuda devices are already restricted, only choose from those
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
# this lets us build cornet-z the same way for all encoder sources
cornet_repo = Path("/zpool/vladlab/active_drive/omaltz/git_repos/CORnet")
import sys
sys.path.insert(0, str(cornet_repo))

import cornet
print("imported cornet from:", cornet.__file__)

# %%
# dataset for the bbox csv
# this version ignores category labels
# it only learns object/no object and where the boxes are

class BBoxOnlyCSVDataset(torch.utils.data.Dataset):
    def __init__(self, images_root, csv_path, transform):
        self.images_root = Path(images_root)
        self.csv_path = Path(csv_path)
        self.transform = transform

        # store all boxes by image name
        img_to_boxes = defaultdict(list)
        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                img_name = row["image_file_name"]
                # csv boxes are x, y, width, height in pixel space
                x = float(row["bbox_x"])
                y = float(row["bbox_y"])
                w = float(row["bbox_w"])
                h = float(row["bbox_h"])
                # convert to x1, y1, x2, y2 while still in pixel space
                box = [x, y, x + w, y + h]

                img_to_boxes[img_name].append(box)
        # only keep images that are actually in the folder
        self.items = []
        missing = 0

        for img_name, boxes in img_to_boxes.items():
            img_path = self.images_root / img_name

            if img_path.exists():
                self.items.append((img_path, boxes))
            else:
                missing += 1
        # fail early if the csv and image folder do not match at all
        if len(self.items) == 0:
            raise RuntimeError(f"no matched images found in {self.images_root}")

        if missing:
            print(f"warning: {missing} images from the csv were missing from {self.images_root}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, boxes_list = self.items[idx]
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        boxes = []

        for x1, y1, x2, y2 in boxes_list:
            # normalize boxes by the original image size
            x1 = max(0.0, min(1.0, x1 / orig_w))
            x2 = max(0.0, min(1.0, x2 / orig_w))
            y1 = max(0.0, min(1.0, y1 / orig_h))
            y2 = max(0.0, min(1.0, y2 / orig_h))
            # skip boxes that became invalid for any reason
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
        boxes = torch.tensor(boxes, dtype=torch.float32)
        image = self.transform(image)
        target = {"boxes_xyxy": boxes, "image_path": str(img_path)}
        return image,target


# each image can have a different number of boxes
# this keeps targets as a list 
def detection_collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

# %%
# make the train and validation dataloaders
# train is shuffled, validation is not

train_ds = BBoxOnlyCSVDataset(config["train_images"], config["train_csv"], transform)
val_ds = BBoxOnlyCSVDataset(config["val_images"], config["val_csv"], transform)

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
# the loss and csv use xyxy in a few places

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
    # get overlap between every pair of boxes
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = box_area_xyxy(boxes1)[:, None]
    area2 = box_area_xyxy(boxes2)[None, :]
    union = area1 + area2 - inter
    iou = inter / union.clamp(min=1e-6)
    # enclosing box for giou
    lt_c = torch.min(boxes1[:, None,:2], boxes2[None, :, :2])
    rb_c = torch.max(boxes1[:, None,2:], boxes2[None, :, 2:])
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
        # make a grid of x and y positions
        y_embed = torch.linspace(0, 1, h, device=device).unsqueeze(1).repeat(1, w)
        x_embed = torch.linspace(0, 1, w, device=device).unsqueeze(0).repeat(h, 1)
        # make the sine/cosine frequencies
        dim_t = torch.arange(self.num_pos_feats, device=device, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t

        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=3).flatten(2)

        # combine y and x position information
        pos = torch.cat((pos_y, pos_x), dim=2)

        # return as b, channels, h, w
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

        # this hook grabs the IT activations during the forward pass
        def hook_fn(module, inp, out):
            self.feat = out

        self.model.IT.register_forward_hook(hook_fn)
    def forward(self, x):
        # run cornet normally
        # the hook above saves the IT feature map
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

    # checkpoints can store the cornet model under slightly different prefixes
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

# freeze the encoder before adding the new head
for p in encoder.parameters():
    p.requires_grad = False

encoder.eval()

print("has IT layer:", hasattr(cornet_base, "IT"))
print("encoder source:", config["encoder_source"])

# %%
# individuation head
# the frozen encoder gives IT features
# the head predicts object/no object and boxes

class IndividuationHead(nn.Module):
    def __init__(
        self,
        in_ch,
        num_queries=10,
        d_model=256,
        nhead=8,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
    ):
        super().__init__()

        self.num_queries = num_queries

        # project IT features into the transformer dimension
        self.input_proj = nn.Conv2d(in_ch, d_model, kernel_size=1)

        # add spatial information to the feature tokens
        self.pos_embed = PositionEmbeddingSine(num_pos_feats=d_model // 2)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        # learned object queries
        # each query can predict one object
        self.query_embed = nn.Embedding(num_queries, d_model)

        # objectness head 0 = no object, 1 = object
        self.obj_head = nn.Linear(d_model, 2)
        # box head: predicts cx, cy, w, h
        self.box_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 4))

    def forward(self, feats):
        src = self.input_proj(feats)
        pos = self.pos_embed(src)
        b, d, h, w = src.shape
        # flatten the feature map into transformer tokens
        src_tokens = src.flatten(2).permute(0, 2, 1)
        pos_tokens = pos.flatten(2).permute(0, 2, 1)

        # repeat the learned queries for each image in the batch
        q = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)
        # queries attend to the IT feature tokens
        hs = self.decoder(tgt=q, memory=src_tokens + pos_tokens)
        obj_logits = self.obj_head(hs)
        boxes = torch.sigmoid(self.box_head(hs))

        return {"pred_obj_logits": obj_logits, "pred_boxes": boxes}


class EncoderWithIndividuationHead(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x):
        feats = self.encoder(x)

        return self.head(feats)

# %%
# build the full model
# first get the IT feature shape, then make the individuation head

with torch.no_grad():
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    feats = encoder(dummy)
    in_ch = feats.shape[1]

print("encoder feature shape:", feats.shape)
print("head input channels:", in_ch)
head = IndividuationHead(
    in_ch=in_ch,
    num_queries=config["num_queries"],
    d_model=config["d_model"],
    nhead=config["nhead"],
    num_decoder_layers=config["num_decoder_layers"],
    dim_feedforward=config["dim_feedforward"],
    dropout=config["dropout"],
).to(device)

model = EncoderWithIndividuationHead(encoder, head).to(device)

# make sure only the new head trains
for p in model.encoder.parameters():
    p.requires_grad = False
for p in model.head.parameters():
    p.requires_grad = True
print(model)

# %%
# match predicted queries to real objects
# this version uses object being there and box quality for matching

def hungarian_matcher_box_obj(pred_obj_logits, pred_boxes, tgt_boxes_xyxy, cost_obj=1.0, cost_bbox=5.0, cost_giou=2.0):
    q, _ = pred_obj_logits.shape
    n = tgt_boxes_xyxy.shape[0]

    if n == 0:
        return torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=torch.long)
    # objectness cost
    # lower cost means higher probability of object
    prob_obj = pred_obj_logits.softmax(-1)[:, 1]
    cost_obj_matrix = -prob_obj[:, None].expand(q, n)

    # box distance cost
    pred_xyxy = cxcywh_to_xyxy(pred_boxes).clamp(0, 1)
    cost_l1 = torch.cdist(pred_xyxy, tgt_boxes_xyxy, p=1)
    # overlap cost
    giou = generalized_box_iou_xyxy(pred_xyxy, tgt_boxes_xyxy)
    cost_giou_matrix = -giou
    # final matching cost
    cost = cost_obj * cost_obj_matrix + cost_bbox * cost_l1 + cost_giou * cost_giou_matrix
    cost = cost.detach().cpu().numpy()

    # scipy does the hungarian matching
    pred_idx, tgt_idx = linear_sum_assignment(cost)
    return torch.as_tensor(pred_idx, dtype=torch.long), torch.as_tensor(tgt_idx, dtype=torch.long)

# %%
# set loss for the individuation head
# first match predictions to targets, then calculate objectness and box losses

class SetCriterionObjBoxes(nn.Module):
    def __init__(
        self,
        no_object_weight=0.1,
        cost_obj=1.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        loss_bbox=5.0,
        loss_giou=2.0,
    ):
        super().__init__()

        # costs are used for matching
        self.cost_obj = cost_obj
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

        # weights are used for the final loss
        self.loss_bbox = loss_bbox
        self.loss_giou = loss_giou

        # downweight no-object so it does not dominate training
        weight = torch.ones(2)
        weight[0] = no_object_weight
        self.register_buffer("ce_weight", weight)

    def forward(self, outputs, targets):
        pred_obj_logits = outputs["pred_obj_logits"]
        pred_boxes = outputs["pred_boxes"]

        b, q, _ = pred_obj_logits.shape

        total_obj = 0.0
        total_l1 = 0.0
        total_giou = 0.0
        n_targets = 0

        for i in range(b):
            tgt_xyxy = targets[i]["boxes_xyxy"]
            n_targets += tgt_xyxy.shape[0]
            # match queries to boxes for this image
            pred_idx, tgt_idx = hungarian_matcher_box_obj(
                pred_obj_logits[i],
                pred_boxes[i],
                tgt_xyxy,
                cost_obj=self.cost_obj,
                cost_bbox=self.cost_bbox,
                cost_giou=self.cost_giou,
            )
            # start with all queriesas no-object
            obj_tgt = torch.zeros((q,), dtype=torch.long, device=pred_obj_logits.device)
            # matched queries are object
            if pred_idx.numel() > 0:
                obj_tgt[pred_idx.to(pred_obj_logits.device)] = 1

            loss_obj = F.cross_entropy(pred_obj_logits[i], obj_tgt, weight=self.ce_weight)
            total_obj += loss_obj

            # box loss only applies to matched predictions
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
            "loss_obj": total_obj / denom,
            "loss_bbox": total_l1 / denom,
            "loss_giou": total_giou / denom,
        }
        losses["loss_total"] = losses["loss_obj"] + self.loss_bbox * losses["loss_bbox"] + self.loss_giou * losses["loss_giou"]
        losses["n_targets"] = n_targets
        return losses

# %%
# set up loss and optimizer
# only the individuation head parameters are updated

criterion = SetCriterionObjBoxes(
    no_object_weight=config["no_object_weight"],
    cost_obj=config["cost_obj"],
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
    # keep the encoder frozen and in eval mode
    model.encoder.eval()

    meters = {"loss_total": [], "loss_obj": [], "loss_bbox": [], "loss_giou": []}

    for it, (images, targets) in enumerate(loader):
        images = torch.stack(images, dim=0).to(device)
        # move target boxes to the gpu
        target_list = []
        for t in targets:
            target_list.append({
                "boxes_xyxy": t["boxes_xyxy"].to(device),
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
                f"obj={np.mean(meters['loss_obj'][-print_every:]):.4f} "
                f"bbox={np.mean(meters['loss_bbox'][-print_every:]):.4f} "
                f"giou={np.mean(meters['loss_giou'][-print_every:]):.4f}"
            )
    return {key: float(np.mean(values)) for key, values in meters.items()}

# %%
# train the individuation head
# the cornet-z IT encoder stays frozen the whole time

history = []
for epoch in range(start_epoch, config["epochs"]):
    print("\n" + "=" * 100)
    print(f"epoch {epoch}/{config['epochs'] - 1}")

    train_stats = run_epoch(model, criterion, train_loader, optimizer, device, train=True, print_every=50)
    val_stats = run_epoch(model, criterion, val_loader, optimizer, device, train=False, print_every=50)

    print("train:", train_stats)
    print("val:  ", val_stats)
    # save the best model based on validation loss
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
    }

    torch.save(ckpt, outdir / "latest.pth.tar")
    torch.save(ckpt, outdir / f"epoch_{epoch:03d}.pth.tar")
    if is_best:
        torch.save(ckpt, outdir / "best.pth.tar")
        print(f"saved best checkpoint: {outdir / 'best.pth.tar'}")

    history.append({
        "epoch": epoch,
        **{f"train_{k}": v for k, v in train_stats.items()},
        **{f"val_{k}": v for k, v in val_stats.items()},
    })
    pd.DataFrame(history).to_csv(outdir / "training_history.csv", index=False)

print("done training")

# %%
# run the test set and save the image/box csv
# each row is one predicted object box
# if the model predicts no objects, the image gets a no_object row
valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
image_folder = Path(config["test_images"])

# use rglob in case the test images are organized in subfolders
image_paths = sorted([p for p in image_folder.rglob("*") if p.suffix.lower() in valid_exts])

print("found test images:", len(image_paths))
print("output csv:", out_csv)

rows = []
model.eval()
object_class = 1

with torch.no_grad():
    for i, image_path in enumerate(image_paths, start=1):
        image = Image.open(image_path).convert("RGB")
        x = transform(image).unsqueeze(0).to(device)

        outputs = model(x)

        obj_logits = outputs["pred_obj_logits"][0]
        pred_boxes = outputs["pred_boxes"][0]

        probs = F.softmax(obj_logits, dim=-1)
        scores, labels = probs.max(dim=-1)

        keep = []
        # keep object predictions above threshold
        for j in range(len(labels)):
            label_idx = labels[j].item()
            score = scores[j].item()

            if label_idx != object_class:
                continue
            if score < config["conf_threshold"]:
                continue
            keep.append((score, j))

        # keep only the top predicted boxes
        keep = sorted(keep, reverse=True)[:config["top_k"]]

        if len(keep) == 0:
            rows.append({
                "image_file_name": image_path.name,
                "x1": "no_object",
                "y1": "no_object",
                "x2": "no_object",
                "y2": "no_object",
            })
        else:
            for score, j in keep:
                box = cxcywh_to_xyxy(pred_boxes[j]).detach().cpu()
                rows.append({
                    "image_file_name": image_path.name,
                    "x1": box[0].item(),
                    "y1": box[1].item(),
                    "x2": box[2].item(),
                    "y2": box[3].item(),
                })

        if i % 50 == 0 or i == len(image_paths):
            print(f"processed {i}/{len(image_paths)} images")
pred_df = pd.DataFrame(rows, columns=["image_file_name", "x1", "y1", "x2", "y2"])
pred_df.to_csv(out_csv, index=False)

print("saved csv to:", out_csv)
print("number of rows:", len(pred_df))
pred_df.head(20)


