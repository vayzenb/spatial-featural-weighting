'''
Train CORnet-Z on the COCO individuation task, using only bounding box information and no category labels.
The model learns to predict how many objects are in the image and where they are, but not what they are.
'''

# %%
import os
import time
import csv
import json
import io
import shlex
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
# put all of the main settings in one place
# this should match the training settings from the cornet-z identification model

config = {
    # data paths
    "train_images": "/zpool/vladlab/data_drive/stimulus_sets/geogaze_COCO_stim/coco_working/working_v3/train_working3",
    "train_csv": "/zpool/vladlab/data_drive/stimulus_sets/geogaze_COCO_stim/coco_working/working_v3/instances_train_filtered3_bboxes.csv",
    "val_images": "/zpool/vladlab/data_drive/stimulus_sets/geogaze_COCO_stim/coco_working/working_v3/val_working3",
    "val_csv": "/zpool/vladlab/data_drive/stimulus_sets/geogaze_COCO_stim/coco_working/working_v3/instances_val_filtered3_bboxes.csv",

    # where to save the model checkpoints
    "output_path": "/zpool/vladlab/data_drive/geogaze_data/cornet_coco_bboxes/cornetz/individuation_critical",

    # task settings
    "num_queries": 10,

    # transformer head settings
    "d_model": 256,
    "nhead": 8,
    "num_decoder_layers": 4,
    "dim_feedforward": 1024,
    "dropout": 0.1,

    # training settings
    "epochs": 200,
    "batch_size": 16,
    "workers": 4,
    "lr": 1e-4,
    "backbone_lr": 1e-4,
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
}


# %%

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
# set the gpu before making the model
if config["ngpus"] > 0:
    set_gpus(config["ngpus"])

# use cuda if we asked for a gpu and one is available
device = torch.device("cuda" if torch.cuda.is_available() and config["ngpus"] > 0 else "cpu")

print("device:", device)
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)

# %%
# box helper functions
# the model predicts boxes as cx, cy, w, h
# the dataset gives boxes as xyxy
# the loss needs both formats, so these functions convert between them

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
    # calculate generalized iou between every box in boxes1 and every box in boxes2
    # boxes should already be in xyxy format
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = box_area_xyxy(boxes1)[:, None]
    area2 = box_area_xyxy(boxes2)[None, :]

    union = area1 + area2 - inter
    iou = inter / union.clamp(min=1e-6)
    # smallest box that contains both boxes
    lt_c = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_c = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])

    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[..., 0] * wh_c[..., 1]
    # giou penalizes boxes that are far apart
    giou = iou - (area_c - union) / area_c.clamp(min=1e-6)
    return giou

# %%
# dataset for the bbox csv
# this version does not use category labels
# it only learns whether there is an boject and where the object is

class BBoxOnlyCSVDataset(torch.utils.data.Dataset):
    def __init__(self, images_root, csv_path, transform):
        self.images_root = Path(images_root)
        self.csv_path = Path(csv_path)
        self.transform = transform
        # save all boxes for each image
        img_to_boxes = defaultdict(list)
        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                img_name = row["image_file_name"]

                # csv boxes are x, y, width, height
                x = float(row["bbox_x"])
                y = float(row["bbox_y"])
                w = float(row["bbox_w"])
                h = float(row["bbox_h"])

                # model loss wants x1, y1, x2, y2
                box = [x, y, x + w, y + h]
                img_to_boxes[img_name].append(box)
        # only keep images that are actually in the image folder
        self.items = []
        missing = 0

        for img_name, boxes in img_to_boxes.items():
            img_path = self.images_root / img_name

            if img_path.exists():
                self.items.append((img_path, boxes))
            else:
                missing += 1
        # stop early if something is wrong with the paths
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
        return image, target

# each image can have a different number of boxes
# this keeps the images and targets as lists instead of forcing them into one tensor
def detection_collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

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
# set up the output folder
# this is where checkpoints and the config will be saved

outdir = Path(config["output_path"])
outdir.mkdir(parents=True, exist_ok=True)
config_path = outdir / "config.json"
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
print("saved config to:", config_path)

# %%
# make the train and validation datasets
# these read the csv files and load the matching images

train_ds = BBoxOnlyCSVDataset(config["train_images"], config["train_csv"], transform)
val_ds = BBoxOnlyCSVDataset(config["val_images"], config["val_csv"], transform)
# make the dataloaders
# shuffle train so the model sees images in a different order each epoch
# do not shuffle validation so it stays consistent

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
# quick check that the datasets loaded correctly
print("number of training images:", len(train_ds))
print("number of validation images:", len(val_ds))

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
        # make a grid of y and x positions from 0 to 1
        y_embed = torch.linspace(0, 1, h, device=device).unsqueeze(1).repeat(1, w)
        x_embed = torch.linspace(0, 1, w, device=device).unsqueeze(0).repeat(h, 1)
        # make different frequencies for the sine/cosine position values
        dim_t = torch.arange(self.num_pos_feats, device=device, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        # turn each x/y location into a vector of sine/cosine values
        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t

        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=3).flatten(2)

        # combine y and x position information
        pos = torch.cat((pos_y, pos_x), dim=2)
        # put it in the same shape as a feature map: b, channels, h, w
        pos = pos.permute(2, 0, 1).unsqueeze(0).repeat(b, 1, 1, 1)
        return pos

# %%
# point python to the local cornet repo
# this lets us import the lab's cornet code

cornet_repo = Path("/zpool/vladlab/active_drive/omaltz/git_repos/CORnet")
import sys
sys.path.insert(0, str(cornet_repo))
import cornet
print("imported cornet from:", cornet.__file__)

# %%
# set up cornet-z
# no model selection here because this notebook only trains cornet-z

cornet_base = cornet.cornet_z(pretrained=False, map_location=device)
# unwrap if cornet gives back a dataparallel model
if hasattr(cornet_base, "module"):
    cornet_base = cornet_base.module
# sanity check that the model has the layers we expect
print("cornet base type:", type(cornet_base))
print("has IT layer:", hasattr(cornet_base, "IT"))
print("has decoder:", hasattr(cornet_base, "decoder"))

# %%
# cornet-z feature extractor
# this runs the full cornet model, but saves the IT layer output with a hook

class CornetITSpatialBackbone(nn.Module):
    def __init__(self, cornet_model):
        super().__init__()
        self.model = cornet_model
        self.feat = None

        # sanity check that cornet-z has the IT layer
        if not hasattr(self.model, "IT"):
            raise ValueError("cornet model does not have an IT layer")
        # this hook saves the IT activations during the forward pass
        def hook_fn(module, inp, out):
            self.feat = out
        self.model.IT.register_forward_hook(hook_fn)

    def forward(self, x):
        # run cornet normally
        # the hook above grabs the IT feature map while it runs
        _ = self.model(x)
        if self.feat is None:
            raise RuntimeError("IT hook did not capture an output")
        return self.feat

# %%
# full individuation model
# cornet-z gives us the IT feature map
# the transformer head predicts object/no boject and boxes

class CornetITDETRObjness(nn.Module):
    def __init__(
        self,
        cornet_base,
        num_queries=10,
        d_model=256,
        nhead=8,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.num_queries = num_queries

        # get the IT feature map from cornet-z
        self.backbone = CornetITSpatialBackbone(cornet_base)

        # this gets made during the first forward pass
        # we need to see the IT feature size before making the projection layer
        self.proj = None
        self.d_model = d_model

        # tells the transformer where each feature-map location came from
        self.pos_embed = PositionEmbeddingSine(num_pos_feats=d_model // 2)

        # transformer decoder
        # the boject queries attend to the IT feature map
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

        # objectness prediction
        # 0 = no boject, 1 = object
        self.obj_head = nn.Linear(d_model, 2)

        # box prediction
        # sigmoid at the end of forward keeps boxes between 0 and 1
        self.box_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 4))

    def _ensure_proj(self, feat):
        # make the 1x1 projection once we know how many IT channels there are
        if self.proj is None:
            in_ch = feat.shape[1]
            self.proj = nn.Conv2d(in_ch, self.d_model, kernel_size=1).to(feat.device)
            self.add_module("input_proj", self.proj)

    def forward(self, x):
        # get IT feature map
        feat = self.backbone(x)
        # project IT features into transformer dimension
        self._ensure_proj(feat)
        src = self.proj(feat)
        # add spatial position information
        pos = self.pos_embed(src)
        b, d, h, w = src.shape
        # flatten feature map into tokens for the transformer
        src_tokens = src.flatten(2).permute(0, 2, 1)
        pos_tokens = pos.flatten(2).permute(0, 2, 1)
        # repeat learned queries for each image in the batch
        q = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)
        # each query attends to the IT feature tokens
        hs = self.decoder(tgt=q, memory=src_tokens + pos_tokens)
        # predict objectness and box for each query
        obj_logits = self.obj_head(hs)
        boxes = torch.sigmoid(self.box_head(hs))
        return {"pred_obj_logits": obj_logits, "pred_boxes": boxes}

# %%
# match predicted queries to real objects

def hungarian_matcher_box_obj(
    pred_obj_logits,
    pred_boxes,
    tgt_boxes_xyxy,
    cost_obj=1.0,
    cost_bbox=5.0,
    cost_giou=2.0,
):
    q, _ = pred_obj_logits.shape
    n = tgt_boxes_xyxy.shape[0]

    # if there are no objects in the image, there is nothing to match
    if n == 0:
        return torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=torch.long)

    # object or no object  cost
    # lower cost means the query gives higher probability to object
    prob_obj = pred_obj_logits.softmax(-1)[:, 1]
    cost_obj_matrix = -prob_obj[:, None].expand(q, n)

    # box cost
    # convert predicted boxes to xyxy so they match the target box format
    pred_xyxy = cxcywh_to_xyxy(pred_boxes).clamp(0, 1)
    cost_l1 = torch.cdist(pred_xyxy, tgt_boxes_xyxy, p=1)

    # giou cost
    # better overlap should give a lower matching cost
    giou = generalized_box_iou_xyxy(pred_xyxy, tgt_boxes_xyxy)
    cost_giou_matrix = -giou

    # final matching cost
    cost = cost_obj * cost_obj_matrix + cost_bbox * cost_l1 + cost_giou * cost_giou_matrix

    # scipy does the actual hungarian matching
    cost = cost.detach().cpu().numpy()
    pred_idx, tgt_idx = linear_sum_assignment(cost)

    return torch.as_tensor(pred_idx, dtype=torch.long), torch.as_tensor(tgt_idx, dtype=torch.long)

# %%
# set loss for the individuation head
# first match predictions to targets, then calculate object is there and box losses

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

            # match predicted queries to the real boxes in this image
            pred_idx, tgt_idx = hungarian_matcher_box_obj(
                pred_obj_logits[i],
                pred_boxes[i],
                tgt_xyxy,
                cost_obj=self.cost_obj,
                cost_bbox=self.cost_bbox,
                cost_giou=self.cost_giou,
            )

            # start by saying every query predicted no object
            obj_tgt = torch.zeros((q,), dtype=torch.long, device=pred_obj_logits.device)
            # then matched queries are counted as objects
            if pred_idx.numel() > 0:
                obj_tgt[pred_idx.to(pred_obj_logits.device)] = 1
            # objectness loss is calculated for every query
            loss_obj = F.cross_entropy(pred_obj_logits[i], obj_tgt, weight=self.ce_weight)
            total_obj += loss_obj

            # box loss is only calculated for matched predictions
            if pred_idx.numel() > 0:
                p_boxes = pred_boxes[i, pred_idx.to(pred_boxes.device)]
                p_xyxy = cxcywh_to_xyxy(p_boxes).clamp(0, 1)
                t_xyxy = tgt_xyxy[tgt_idx].to(pred_boxes.device)
                l1 = F.l1_loss(p_xyxy, t_xyxy, reduction="mean")
                giou = generalized_box_iou_xyxy(p_xyxy, t_xyxy).diag()
                giou_loss = (1.0 - giou).mean()

                total_l1 += l1
                total_giou += giou_loss

        # average losses over the batch
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
# run one full pass through either the train or validation set
# train=True updates the model
# train=False just checks the loss

def run_epoch(model, criterion, loader, optimizer, device, train=True, print_every=50):
    model.train() if train else model.eval()
    meters = {"loss_total": [], "loss_obj": [], "loss_bbox": [], "loss_giou": []}
    t0 = time.time()
    for it, (images, targets) in enumerate(loader):
        images = torch.stack(images, dim=0).to(device)
        # move the target boxes to the gpu
        target_list = []
        for t in targets:
            target_list.append({
                "boxes_xyxy": t["boxes_xyxy"].to(device),
                "image_path": t.get("image_path", ""),
            })

        # only track gradients during training
        with torch.set_grad_enabled(train):
            outputs = model(images)
            losses = criterion(outputs, target_list)

            if train:
                optimizer.zero_grad(set_to_none=True)
                losses["loss_total"].backward()
                optimizer.step()
        # save losses so we can average them later
        for key in meters:
            meters[key].append(float(losses[key].detach().cpu()))
        # print progress every so often
        if (it + 1) % print_every == 0:
            dt = time.time() - t0
            split = "train" if train else "val"

            print(
                f"{split} iter {it + 1}/{len(loader)} "
                f"loss={np.mean(meters['loss_total'][-print_every:]):.4f} "
                f"obj={np.mean(meters['loss_obj'][-print_every:]):.4f} "
                f"bbox={np.mean(meters['loss_bbox'][-print_every:]):.4f} "
                f"giou={np.mean(meters['loss_giou'][-print_every:]):.4f} "
                f"time={dt:.1f}s"
            )
            t0 = time.time()
    # return average loss for the epoch
    return {key: float(np.mean(values)) for key, values in meters.items()}

# %%
# build the full model and move it to the gpu
# cornet-z gives the IT features, then the transformer head predicts objectness and boxes

model = CornetITDETRObjness(
    cornet_base=cornet_base,
    num_queries=config["num_queries"],
    d_model=config["d_model"],
    nhead=config["nhead"],
    num_decoder_layers=config["num_decoder_layers"],
    dim_feedforward=config["dim_feedforward"],
    dropout=config["dropout"],
).to(device)
# print the model for sanity
print(model)

# %%
# run one dummy batch so the projection layer gets created
# this is needed before setting up the optimizer

images, _ = next(iter(train_loader))
images = torch.stack(images, dim=0).to(device)
with torch.no_grad():
    _ = model(images)

# %%
# set up the loss
# this does matching, objectness loss, bbox loss, and giou loss

criterion = SetCriterionObjBoxes(
    no_object_weight=config["no_object_weight"],
    cost_obj=config["cost_obj"],
    cost_bbox=config["cost_bbox"],
    cost_giou=config["cost_giou"],
    loss_bbox=config["loss_bbox"],
    loss_giou=config["loss_giou"],
).to(device)

# %%
# set up optimizer
# use one learning rate for cornet-z and one for the transformer head

backbone_params = []
head_params = []

for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if name.startswith("backbone."):
        backbone_params.append(p)
    else:
        head_params.append(p)
optimizer = torch.optim.AdamW(
    [
        {"params": backbone_params, "lr": config["backbone_lr"]},
        {"params": head_params, "lr": config["lr"]},
    ],
    weight_decay=config["weight_decay"],
)

start_epoch = 0
best_val = float("inf")
print("backbone params:", len(backbone_params))
print("head params:", len(head_params))

# %%
# resume training from a saved checkpoint
# leave config["resume"] as none to start from scratch

if config["resume"] is not None:
    ckpt = torch.load(config["resume"], map_location=device)

    model.load_state_dict(ckpt["model"], strict=False)
    optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_val = ckpt.get("best_val", best_val)
    print(f"resumed from {config['resume']} at epoch {start_epoch}")
else:
    print("starting from scratch")

# %%
# run training
# each epoch does one train pass, one validation pass, and then saves checkpoints
for epoch in range(start_epoch, config["epochs"]):
    print("\n" + "=" * 100)
    print(f"epoch {epoch}/{config['epochs'] - 1}")
    # train for one epoch
    train_stats = run_epoch(
        model=model,
        criterion=criterion,
        loader=train_loader,
        optimizer=optimizer,
        device=device,
        train=True,
        print_every=50,
    )

    # check validation loss after training
    val_stats = run_epoch(
        model=model,
        criterion=criterion,
        loader=val_loader,
        optimizer=optimizer,
        device=device,
        train=False,
        print_every=50,
    )
    print("train:", train_stats)
    print("val:  ", val_stats)
    # track the best validation loss so far
    is_best = val_stats["loss_total"] < best_val
    if is_best:
        best_val = val_stats["loss_total"]
    # save everything needed to restart training later
    ckpt = {
        "epoch": epoch,
        "best_val": best_val,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "train_stats": train_stats,
        "val_stats": val_stats,
    }

    latest_path = outdir / "latest.pth.tar"
    epoch_path = outdir / f"epoch_{epoch:03d}.pth.tar"
    # always save latest and this epoch
    torch.save(ckpt, latest_path)
    torch.save(ckpt, epoch_path)

    print(f"saved epoch checkpoint: {epoch_path}")

    # also save a separate copy if this is the best model so far
    if is_best:
        best_path = outdir / "best.pth.tar"
        torch.save(ckpt, best_path)
        print(f"saved best checkpoint: {best_path}, val loss = {best_val:.4f}")

print("done")


