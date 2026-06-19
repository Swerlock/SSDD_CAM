#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv5 + SSDD XAI Final Pipeline
================================

Birleştirilmiş final kod:
- Eski kodun güçlü tarafları:
  * Per-detection CAM target
  * Detection IoU summary
  * CAM-IoU ve size-wise small/medium/large analiz
  * selected layers, detect_head_m0/m1/m2, last_conv desteği
  * Grad-CAM / Grad-CAM++ / LayerCAM / EigenCAM
- Yeni eklenen final tez parçaları:
  * Otsu eşikleme
  * Otsu binary mask kaydı
  * M1: CAM bölgesini silme
  * M2: yalnızca CAM bölgesini koruma
  * val.py ile baseline / M1 / M2 mAP-drop
  * M1/M2 panel görselleri

Önerilen çalışma sırası:
1) İlk koşuda:
   RUN_CAM_IOU_ANALYSIS = True
   RUN_M1M2_MAPDROP    = False
   N_IMAGES = 20
2) Görseller/CSV doğruysa:
   RUN_M1M2_MAPDROP = True
   N_IMAGES = None

Colab:
    %cd /content/yolov5
    !python /content/drive/MyDrive/xai_cam_otsu_m1m2_final_merged.py
"""

import os
import sys
import re
import glob
import csv
import gc
import cv2
import yaml
import shutil
import subprocess
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ============================================================
# CONFIGURATION
# ============================================================

YOLO_ROOT = "/content/yolov5"
sys.path.insert(0, YOLO_ROOT)

from models.common import DetectMultiBackend
from utils.general import non_max_suppression

WEIGHTS = "/content/drive/MyDrive/SSDD_A100_Egitim_Sonuclari/YOLOv5l_Orijinal_Veri5/weights/best.pt"
IMG_DIR = "/content/SSDD_local/SSDD_YOLO/images/test"
LABEL_DIR = "/content/SSDD_local/SSDD_YOLO/labels/test"
OUT_DIR = "/content/drive/MyDrive/SSDD_XAI_FINAL_1024"
VAL_SCRIPT = os.path.join(YOLO_ROOT, "val.py")

IMG_SIZE = 1024
BATCH_SIZE = 16
DEVICE_ID = "0"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["ship"]
NC = 1

# -------------------------
# Run flags
# -------------------------
RUN_CAM_IOU_ANALYSIS = True       # Detection IoU + CAM-IoU + Otsu mask summary
RUN_SAVE_VISUALS = True           # Overlay / panel / Otsu mask kaydı
RUN_M1M2_MAPDROP = True     # Ağır aşama: M1/M2 dataset + val.py

# İlk deneme için 20 önerilir. Finalde None yap.
N_IMAGES = None # None

# -------------------------
# CAM methods/layers for CAM-IoU analysis
# -------------------------
METHODS = ["gradcam","gradcampp","eigencam","layercam"]

METHOD_LAYER_SPECS = {
    "gradcam": [20, 22, 23,"detect_head_m1", "last_conv"],
    "gradcampp": [ 13, 17, 20, 22, 23,"last_conv"],
    "layercam": [ 9, 13, 17, 20, 22, 23,"last_conv"],
    "eigencam": [23,"last_conv"],
}

# M1/M2 mAP-drop için daha dar final aday seti.
# Esra referansı + bizim ön taramada iyi görünen adaylar.
M1M2_LAYER_SPECS = {
    "gradcam": [20,22,23,"detect_head_m1", "last_conv"],
    "gradcampp": [13, 17, 20, 22,23, "last_conv"],
    "layercam": [ 9, 13, 17, 20, 22,23,"last_conv" ],
    "eigencam": [23,"last_conv"],
}

# -------------------------
# Detection/CAM settings
# -------------------------
CONF_THRES = 0.25
NMS_IOU_THRES = 0.45
MAX_DETECTIONS_PER_IMAGE = 50

USE_PER_DETECTION_AGGREGATION = False
AGGREGATION_MODE = "mean"  # "max" or "mean"

# Fixed thresholds + Otsu birlikte raporlanacak.
FIXED_CAM_THRESHOLDS = [0.30, 0.40, 0.50, 0.60]
CAM_MIN_AREA = 40

# Size-wise GT bbox area ratio thresholds
SMALL_AREA_RATIO = 0.002
LARGE_AREA_RATIO = 0.020

# Grad-CAM++ global fallback target
LSE_TAU = 0.25

# Visualization
ALPHA = 0.55
VIS_FIXED_THRESHOLD = 0.50
SAVE_OVERLAY_ONLY = True
SAVE_PANEL = True
SAVE_OTSU_MASK = True
SAVE_M1M2_SAMPLE_PANELS = True
M1M2_FILL_MODE = "mean"  # "black" or "mean"

PRINT_PROGRESS_EVERY = 10
PRINT_EVERY_IMAGE_IOU = True

ANSI_ESCAPE = re.compile(r"\x1B\[[0-9;]*[JKmsu]")
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

# CSV paths
DETECTION_IOU_CSV_PATH = os.path.join(OUT_DIR, "csv", "detection_iou_per_image.csv")
DETECTION_IOU_SUMMARY_CSV_PATH = os.path.join(OUT_DIR, "csv", "detection_iou_summary.csv")
CAM_OBJECT_IOU_CSV_PATH = os.path.join(OUT_DIR, "csv", "cam_iou_per_object_layer.csv")
CAM_SIZE_SUMMARY_CSV_PATH = os.path.join(OUT_DIR, "csv", "cam_iou_sizewise_summary.csv")
M1M2_MAPDROP_CSV_PATH = os.path.join(OUT_DIR, "csv", "m1_m2_mapdrop_summary.csv")

# ============================================================
# ACTIVATION / GRADIENT HOOKS
# ============================================================

class ActivationsAndGradients:
    def __init__(self, model, target_layers, detach=True):
        self.model = model
        self.gradients = []
        self.activations = []
        self.handles = []
        self.detach = detach
        for target_layer in target_layers:
            self.handles.append(target_layer.register_forward_hook(self.save_activation))

    def save_activation(self, module, input, output):
        activation = output
        if isinstance(output, (list, tuple)):
            activation = None
            for item in output:
                if isinstance(item, torch.Tensor) and item.dim() == 4:
                    activation = item
                    break
        if activation is None or not isinstance(activation, torch.Tensor):
            return
        if activation.requires_grad:
            activation.register_hook(self.save_gradient)
        self.activations.append(activation.detach().cpu() if self.detach else activation)

    def save_gradient(self, grad):
        self.gradients.append(grad.detach().cpu() if self.detach else grad)

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.handles = []


class ActivationsOnly:
    def __init__(self, model, target_layers):
        self.model = model
        self.activations = []
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(target_layer.register_forward_hook(self.save_activation))

    def save_activation(self, module, input, output):
        activation = output
        if isinstance(output, (list, tuple)):
            activation = None
            for item in output:
                if isinstance(item, torch.Tensor) and item.dim() == 4:
                    activation = item
                    break
        if isinstance(activation, torch.Tensor):
            self.activations.append(activation.detach().cpu())

    def __call__(self, x):
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.handles = []


# ============================================================
# BASE CAM + METHODS
# ============================================================

class BaseCAM:
    def __init__(self, model: nn.Module, target_layers: List[nn.Module], uses_gradients=True, detach=True):
        self.model = model.eval()
        self.target_layers = target_layers
        self.uses_gradients = uses_gradients
        self.detach = detach
        self.device = next(self.model.parameters()).device
        if self.uses_gradients:
            self.activations_and_grads = ActivationsAndGradients(self.model, target_layers, detach=self.detach)
        else:
            self.activations_only = ActivationsOnly(self.model, target_layers)

    def get_cam_weights(self, input_tensor, target_layer, targets, activations, grads):
        raise NotImplementedError

    def get_cam_image(self, input_tensor, target_layer, targets, activations, grads):
        weights = self.get_cam_weights(input_tensor, target_layer, targets, activations, grads)
        if isinstance(activations, torch.Tensor):
            activations = activations.detach().cpu().numpy()
        if len(activations.shape) != 4:
            raise ValueError(f"Invalid activation shape: {activations.shape}")
        weighted_activations = weights[:, :, None, None] * activations
        return weighted_activations.sum(axis=1)

    def compute_cam_per_layer(self, input_tensor, targets):
        if self.uses_gradients:
            activations_list = [a.cpu().data.numpy() if isinstance(a, torch.Tensor) else a for a in self.activations_and_grads.activations]
            grads_list = [g.cpu().data.numpy() if isinstance(g, torch.Tensor) else g for g in self.activations_and_grads.gradients]
        else:
            activations_list = [a.cpu().data.numpy() if isinstance(a, torch.Tensor) else a for a in self.activations_only.activations]
            grads_list = [None] * len(activations_list)

        cams = []
        for i, target_layer in enumerate(self.target_layers):
            activations = activations_list[i] if i < len(activations_list) else None
            grads = grads_list[i] if i < len(grads_list) else None
            if activations is None:
                raise RuntimeError(f"Activation is None for layer {target_layer}")
            if self.uses_gradients and grads is None:
                raise RuntimeError(f"Gradient is None for layer {target_layer}")
            cam = self.get_cam_image(input_tensor, target_layer, targets, activations, grads)
            cam = np.maximum(cam, 0)
            cams.append(cam)
        return cams[0] if len(cams) == 1 else np.mean(cams, axis=0)

    def forward(self, input_tensor, targets=None):
        input_tensor = input_tensor.detach().clone().to(self.device)
        if self.uses_gradients:
            input_tensor.requires_grad_(True)
            outputs = self.activations_and_grads(input_tensor)
            self.model.zero_grad(set_to_none=True)
            loss = targets[0](outputs)
            loss.backward(retain_graph=True)
            return self.compute_cam_per_layer(input_tensor, targets)
        else:
            with torch.no_grad():
                _ = self.activations_only(input_tensor)
            return self.compute_cam_per_layer(input_tensor, targets)

    def __call__(self, input_tensor, targets=None):
        return self.forward(input_tensor, targets)

    def release(self):
        if self.uses_gradients:
            self.activations_and_grads.release()
        else:
            self.activations_only.release()

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


class GradCAM(BaseCAM):
    def __init__(self, model, target_layers):
        super().__init__(model=model, target_layers=target_layers, uses_gradients=True, detach=True)
    def get_cam_weights(self, input_tensor, target_layer, targets, activations, grads):
        return np.mean(grads, axis=(2, 3))


class GradCAMPlusPlus(BaseCAM):
    def __init__(self, model, target_layers):
        super().__init__(model=model, target_layers=target_layers, uses_gradients=True, detach=True)
    def get_cam_weights(self, input_tensor, target_layer, targets, activations, grads):
        grads_power_2 = grads ** 2
        grads_power_3 = grads_power_2 * grads
        sum_activations = np.sum(activations, axis=(2, 3))
        eps = 1e-7
        aij = grads_power_2 / (2 * grads_power_2 + sum_activations[:, :, None, None] * grads_power_3 + eps)
        aij = np.where(grads != 0, aij, 0)
        return np.sum(np.maximum(grads, 0) * aij, axis=(2, 3))


class LayerCAM(BaseCAM):
    def __init__(self, model, target_layers):
        super().__init__(model=model, target_layers=target_layers, uses_gradients=True, detach=True)
    def get_cam_image(self, input_tensor, target_layer, targets, activations, grads):
        if isinstance(activations, torch.Tensor):
            activations = activations.detach().cpu().numpy()
        if isinstance(grads, torch.Tensor):
            grads = grads.detach().cpu().numpy()
        return (np.maximum(grads, 0) * activations).sum(axis=1)
    def get_cam_weights(self, input_tensor, target_layer, targets, activations, grads):
        return None


def get_2d_projection(activation_batch):
    activation_batch = activation_batch.astype(np.float32)
    activation_batch[np.isnan(activation_batch)] = 0
    b, c, h, w = activation_batch.shape
    projections = []
    for i in range(b):
        reshaped = activation_batch[i].reshape(c, h * w).transpose()
        try:
            _, _, vt = np.linalg.svd(reshaped, full_matrices=False)
            projection = reshaped @ vt[0, :]
            projection = projection.reshape(h, w)
        except Exception:
            projection = np.zeros((h, w), dtype=np.float32)
        projections.append(projection)
    return np.stack(projections, axis=0).astype(np.float32)


class EigenCAM(BaseCAM):
    def __init__(self, model, target_layers):
        super().__init__(model=model, target_layers=target_layers, uses_gradients=False, detach=True)
    def get_cam_image(self, input_tensor, target_layer, targets, activations, grads):
        return get_2d_projection(activations)
    def get_cam_weights(self, input_tensor, target_layer, targets, activations, grads):
        return None


# ============================================================
# YOLO TARGETS
# ============================================================

def safe_sigmoid_if_needed(x):
    if x.max() > 1.0 or x.min() < 0.0:
        return x.sigmoid()
    return x


def xywh2xyxy_torch(x):
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def box_iou_torch(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(0)
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union = area1[:, None] + area2[None, :] - inter + 1e-7
    return inter / union


class YOLOGlobalTarget:
    def __call__(self, model_output):
        pred = model_output[0] if isinstance(model_output, (list, tuple)) else model_output
        obj = safe_sigmoid_if_needed(pred[..., 4])
        cls = safe_sigmoid_if_needed(pred[..., 5:])
        best_cls, _ = cls.max(dim=-1)
        return (obj * best_cls).sum()


class YOLOGlobalTargetLSE:
    def __init__(self, tau=0.25):
        self.tau = float(tau)
    def __call__(self, model_output):
        pred = model_output[0] if isinstance(model_output, (list, tuple)) else model_output
        obj = safe_sigmoid_if_needed(pred[..., 4])
        cls = safe_sigmoid_if_needed(pred[..., 5:])
        best_cls, _ = cls.max(dim=-1)
        score = (obj * best_cls).reshape(-1)
        tau = max(self.tau, 1e-6)
        return tau * torch.logsumexp(score / tau, dim=0)


class YOLODetectionBoxTarget:
    """NMS tespit kutusuna en çok uyan raw YOLO prediction skorunu hedef alır."""
    def __init__(self, det_box_xyxy_640, det_class):
        self.det_box_xyxy_640 = torch.tensor(det_box_xyxy_640, dtype=torch.float32, device=DEVICE).view(1, 4)
        self.det_class = int(det_class)
    def __call__(self, model_output):
        pred = model_output[0] if isinstance(model_output, (list, tuple)) else model_output
        boxes_xywh = pred[..., :4]
        boxes_xyxy = xywh2xyxy_torch(boxes_xywh).reshape(-1, 4)
        obj = safe_sigmoid_if_needed(pred[..., 4]).reshape(-1)
        cls = safe_sigmoid_if_needed(pred[..., 5:])
        if cls.shape[-1] <= self.det_class:
            cls_score = cls.max(dim=-1)[0].reshape(-1)
        else:
            cls_score = cls[..., self.det_class].reshape(-1)
        score = obj * cls_score
        with torch.no_grad():
            ious = box_iou_torch(boxes_xyxy, self.det_box_xyxy_640).squeeze(1)
            selection_score = ious * score.detach()
            best_idx = int(torch.argmax(selection_score).item())
        return score[best_idx]


# ============================================================
# BASIC UTILS
# ============================================================

def disable_inplace(model):
    for m in model.modules():
        if isinstance(m, (nn.ReLU, nn.SiLU, nn.LeakyReLU)) and hasattr(m, "inplace"):
            m.inplace = False
    return model


def normalize01(x):
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x -= x.min()
    mx = x.max()
    if mx > 1e-8:
        x /= mx
    return x


def safe_name(name):
    name = str(name)
    for ch in ["/", "\\", " ", ":", ".", "(", ")", ","]:
        name = name.replace(ch, "_")
    return name


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def cleanup_cuda(full=False):
    if full:
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if full:
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass


def load_image_no_pad(path):
    img0 = cv2.imread(path)
    if img0 is None:
        raise ValueError(f"Cannot read image: {path}")
    h0, w0 = img0.shape[:2]
    img = cv2.resize(img0, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    x = torch.from_numpy(img).float() / 255.0
    return x.unsqueeze(0).to(DEVICE), img0, (h0, w0)


def resize_to_original(cam01, w0, h0):
    cam = cv2.resize(cam01, (w0, h0), interpolation=cv2.INTER_LINEAR)
    return normalize01(cam)


def overlay_cam(img0_bgr, cam01, alpha=0.55):
    heat = cv2.applyColorMap(np.uint8(255 * normalize01(cam01)), cv2.COLORMAP_JET)
    return cv2.addWeighted(img0_bgr, 1 - alpha, heat, alpha, 0)


def otsu_mask(cam01):
    cam01 = normalize01(cam01)
    u8 = np.uint8(np.clip(cam01 * 255, 0, 255))
    thr, mb = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = mb > 0
    return mask.astype(bool), {
        "otsu_threshold": float(thr / 255.0),
        "otsu_mask_area_ratio": float(mask.mean()),
    }


def mask_to_bgr(mask):
    return cv2.cvtColor((mask.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)


def make_m1_m2(img0, mask):
    if M1M2_FILL_MODE == "mean":
        fill = img0.reshape(-1, 3).mean(axis=0).astype(np.uint8)
    else:
        fill = np.array([0, 0, 0], dtype=np.uint8)
    m1 = img0.copy()
    m2 = img0.copy()
    m1[mask] = fill       # CAM bölgesi silinir
    m2[~mask] = fill      # sadece CAM bölgesi kalır
    return m1, m2


# ============================================================
# LAST CONV / LAYER SELECTION
# ============================================================

def find_last_conv(net):
    last = None
    last_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            last = m
            last_name = name
        if hasattr(m, "conv") and isinstance(getattr(m, "conv"), nn.Conv2d):
            last = getattr(m, "conv")
            last_name = name + ".conv"
    return last, last_name


def get_layer_module_and_name(net, spec):
    if isinstance(spec, int):
        if spec >= len(net.model):
            raise ValueError(f"Layer index {spec} does not exist in net.model")
        module = net.model[spec]
        name = f"layer_{spec}_{module.__class__.__name__}"
        return module, safe_name(name)
    if isinstance(spec, str) and spec.startswith("detect_head_m"):
        idx = int(spec.replace("detect_head_m", ""))
        detect_module = net.model[-1]
        if not hasattr(detect_module, "m"):
            raise ValueError("Detect module has no attribute m")
        if idx >= len(detect_module.m):
            raise ValueError(f"Detect head m{idx} does not exist")
        module = detect_module.m[idx]
        name = f"detect_head_m{idx}"
        return module, safe_name(name)
    if isinstance(spec, str) and spec == "last_conv":
        module, last_name = find_last_conv(net)
        if module is None:
            raise ValueError("No last Conv2d layer found")
        name = f"last_conv_{last_name}"
        return module, safe_name(name)
    raise ValueError(f"Unsupported layer spec: {spec}")


def build_method_layers(net, layer_specs):
    method_layers = {}
    for method_name, specs in layer_specs.items():
        method_layers[method_name] = []
        for spec in specs:
            module, layer_name = get_layer_module_and_name(net, spec)
            method_layers[method_name].append({"spec": spec, "module": module, "name": layer_name})
    return method_layers


# ============================================================
# LABELS / IOU
# ============================================================

def load_yolo_labels(label_path, w0, h0):
    boxes, classes = [], []
    if not os.path.exists(label_path):
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    with open(label_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc = float(parts[1]) * w0; yc = float(parts[2]) * h0
        bw = float(parts[3]) * w0; bh = float(parts[4]) * h0
        boxes.append([xc-bw/2, yc-bh/2, xc+bw/2, yc+bh/2])
        classes.append(cls)
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    return np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.int32)


def box_iou_np(boxes1, boxes2):
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    x11, y11, x12, y12 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    x21, y21, x22, y22 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]
    inter_x1 = np.maximum(x11[:, None], x21[None, :])
    inter_y1 = np.maximum(y11[:, None], y21[None, :])
    inter_x2 = np.minimum(x12[:, None], x22[None, :])
    inter_y2 = np.minimum(y12[:, None], y22[None, :])
    inter_area = np.maximum(inter_x2-inter_x1, 0) * np.maximum(inter_y2-inter_y1, 0)
    area1 = np.maximum(x12-x11, 0) * np.maximum(y12-y11, 0)
    area2 = np.maximum(x22-x21, 0) * np.maximum(y22-y21, 0)
    union = area1[:, None] + area2[None, :] - inter_area + 1e-7
    return inter_area / union


def get_size_class(box, w0, h0):
    area = max(box[2]-box[0], 0) * max(box[3]-box[1], 0)
    area_ratio = area / (w0*h0 + 1e-7)
    if area_ratio < SMALL_AREA_RATIO:
        return "small", float(area_ratio)
    if area_ratio >= LARGE_AREA_RATIO:
        return "large", float(area_ratio)
    return "medium", float(area_ratio)


# ============================================================
# DETECTION
# ============================================================

def get_detections(model, x):
    with torch.no_grad():
        pred = model(x)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        det = non_max_suppression(pred, conf_thres=CONF_THRES, iou_thres=NMS_IOU_THRES, max_det=MAX_DETECTIONS_PER_IMAGE)[0]
    if det is None or len(det) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    return det.detach().cpu().numpy().astype(np.float32)


def scale_det_to_original(det_640, w0, h0):
    if len(det_640) == 0:
        return det_640.copy()
    det_orig = det_640.copy()
    det_orig[:, [0, 2]] *= (w0 / IMG_SIZE)
    det_orig[:, [1, 3]] *= (h0 / IMG_SIZE)
    det_orig[:, [0, 2]] = np.clip(det_orig[:, [0, 2]], 0, w0)
    det_orig[:, [1, 3]] = np.clip(det_orig[:, [1, 3]], 0, h0)
    return det_orig


def evaluate_detection_iou(image_name, det_orig, gt_boxes, gt_classes):
    if len(det_orig) == 0:
        return {"image": image_name, "gt_count": int(len(gt_boxes)), "pred_count": 0, "best_iou": "", "mean_best_gt_iou": "", "best_conf": "", "best_pred_class": "", "best_gt_class": "", "tp50": 0, "tp75": 0}
    pred_boxes = det_orig[:, :4]
    pred_conf = det_orig[:, 4]
    pred_cls = det_orig[:, 5].astype(np.int32)
    if len(gt_boxes) == 0:
        best_idx = int(np.argmax(pred_conf))
        return {"image": image_name, "gt_count": 0, "pred_count": int(len(pred_boxes)), "best_iou": "", "mean_best_gt_iou": "", "best_conf": float(pred_conf[best_idx]), "best_pred_class": int(pred_cls[best_idx]), "best_gt_class": "", "tp50": 0, "tp75": 0}
    ious = box_iou_np(pred_boxes, gt_boxes)
    best_pred_idx, best_gt_idx = np.unravel_index(np.argmax(ious), ious.shape)
    best_iou = float(ious[best_pred_idx, best_gt_idx])
    mean_best_gt_iou = float(ious.max(axis=0).mean())
    return {"image": image_name, "gt_count": int(len(gt_boxes)), "pred_count": int(len(pred_boxes)), "best_iou": best_iou, "mean_best_gt_iou": mean_best_gt_iou, "best_conf": float(pred_conf[best_pred_idx]), "best_pred_class": int(pred_cls[best_pred_idx]), "best_gt_class": int(gt_classes[best_gt_idx]), "tp50": int(best_iou >= 0.50), "tp75": int(best_iou >= 0.75)}


# ============================================================
# CAM COMPONENTS / CAM-IOU
# ============================================================

def mask_to_component_boxes(mask, min_area=5):
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    boxes, areas = [], []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        boxes.append([x, y, x+w, y+h])
        areas.append(area)
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    return np.array(boxes, dtype=np.float32), np.array(areas, dtype=np.int32)


def cam_to_component_boxes(cam01, threshold=0.50, min_area=5):
    mask = (cam01 >= threshold)
    return mask_to_component_boxes(mask, min_area=min_area)


def compute_cam_object_iou_rows(image_name, method_name, layer_name, cam01, gt_boxes, gt_classes, w0, h0, otsu_info):
    rows = []
    threshold_items = []
    for th in FIXED_CAM_THRESHOLDS:
        threshold_items.append(("fixed", float(th), cam01 >= float(th)))
    otsu_m, _ = otsu_mask(cam01)
    threshold_items.append(("otsu", float(otsu_info["otsu_threshold"]), otsu_m))

    for threshold_type, threshold, mask in threshold_items:
        cam_boxes, cam_areas = mask_to_component_boxes(mask, min_area=CAM_MIN_AREA)
        for obj_idx, gt_box in enumerate(gt_boxes):
            size_class, gt_area_ratio = get_size_class(gt_box, w0, h0)
            base = {"image": image_name, "method": method_name, "layer": layer_name, "threshold_type": threshold_type, "threshold": float(threshold), "object_index": int(obj_idx), "gt_class": int(gt_classes[obj_idx]), "size_class": size_class, "gt_area_ratio": gt_area_ratio, "otsu_threshold": otsu_info["otsu_threshold"], "otsu_mask_area_ratio": otsu_info["otsu_mask_area_ratio"]}
            if len(cam_boxes) == 0:
                base.update({"cam_iou": "", "best_cam_area": "", "hit50": 0, "hit75": 0})
                rows.append(base)
                continue
            ious = box_iou_np(gt_box[None, :].astype(np.float32), cam_boxes)[0]
            best_idx = int(np.argmax(ious))
            best_iou = float(ious[best_idx])
            base.update({"cam_iou": best_iou, "best_cam_area": int(cam_areas[best_idx]), "hit50": int(best_iou >= 0.50), "hit75": int(best_iou >= 0.75)})
            rows.append(base)
    return rows


# ============================================================
# CAM COMPUTATION
# ============================================================

def create_cam_algorithm(method_name, model, layer_module):
    method_name = method_name.lower()
    if method_name == "gradcam":
        return GradCAM(model, [layer_module])
    if method_name == "gradcampp":
        return GradCAMPlusPlus(model, [layer_module])
    if method_name == "layercam":
        return LayerCAM(model, [layer_module])
    if method_name == "eigencam":
        return EigenCAM(model, [layer_module])
    raise ValueError(f"Unknown method: {method_name}")


def create_global_target(method_name):
    return [YOLOGlobalTargetLSE(tau=LSE_TAU)] if method_name.lower() == "gradcampp" else [YOLOGlobalTarget()]


def compute_single_cam(method_name, model, layer_module, x, targets):
    cam_algorithm = create_cam_algorithm(method_name, model, layer_module)
    cam = cam_algorithm(x, targets)
    cam_algorithm.release()
    del cam_algorithm
    cleanup_cuda(False)
    return normalize01(cam[0])


def compute_cam_for_layer(method_name, model, layer_module, x, det_640, w0, h0):
    method_name = method_name.lower()
    if method_name == "eigencam":
        cam = compute_single_cam(method_name, model, layer_module, x, targets=None)
        return resize_to_original(cam, w0, h0)
    if not USE_PER_DETECTION_AGGREGATION or len(det_640) == 0:
        cam = compute_single_cam(method_name, model, layer_module, x, targets=create_global_target(method_name))
        return resize_to_original(cam, w0, h0)
    detection_cams = []
    for det in det_640:
        target = [YOLODetectionBoxTarget(det_box_xyxy_640=det[:4], det_class=int(det[5]))]
        try:
            cam = compute_single_cam(method_name, model, layer_module, x, targets=target)
            cam = resize_to_original(cam, w0, h0)
            detection_cams.append(cam)
        except Exception as e:
            print(f"[PER-DET-CAM-ERROR] Method={method_name} Error={e}")
    if len(detection_cams) == 0:
        cam = compute_single_cam(method_name, model, layer_module, x, targets=create_global_target(method_name))
        return resize_to_original(cam, w0, h0)
    stack = np.stack(detection_cams, axis=0)
    return normalize01(stack.mean(axis=0) if AGGREGATION_MODE == "mean" else stack.max(axis=0))


# ============================================================
# VISUALIZATION
# ============================================================

def draw_boxes(img, boxes, color, label_prefix="", thickness=2):
    out = img.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        if label_prefix:
            cv2.putText(out, f"{label_prefix}{i}", (x1, max(y1-5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def add_title(img, title):
    h, w = img.shape[:2]
    banner_h = 34
    banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
    out = np.vstack([banner, img])
    cv2.putText(out, title[:120], (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
    return out


def make_visual_panel(img0, cam01, method_name, layer_name, det_orig, gt_boxes, rows_for_vis, out_path):
    overlay = overlay_cam(img0, cam01, ALPHA)
    otsu_m, otsu_info = otsu_mask(cam01)
    left = draw_boxes(img0, gt_boxes, (0,255,0), "GT", 2)
    left = draw_boxes(left, det_orig[:, :4], (0,0,255), "D", 2)
    cam_boxes, _ = cam_to_component_boxes(cam01, threshold=VIS_FIXED_THRESHOLD, min_area=CAM_MIN_AREA)
    middle = overlay.copy()
    middle = draw_boxes(middle, gt_boxes, (0,255,0), "GT", 2)
    middle = draw_boxes(middle, det_orig[:, :4], (0,0,255), "D", 2)
    middle = draw_boxes(middle, cam_boxes, (255,255,0), "CAM", 1)
    right = overlay.copy()
    valid_ious = [float(r["cam_iou"]) for r in rows_for_vis if r["cam_iou"] != "" and r["threshold_type"] == "fixed" and abs(float(r["threshold"]) - VIS_FIXED_THRESHOLD) < 1e-9]
    mean_cam_iou = float(np.mean(valid_ious)) if valid_ious else 0.0
    left = add_title(left, "GT + Detection BBoxes")
    middle = add_title(middle, "GT + Detection + CAM components")
    right = add_title(right, f"{method_name} | {layer_name} | IoU@{VIS_FIXED_THRESHOLD:.2f}: {mean_cam_iou:.4f} | Otsu:{otsu_info['otsu_threshold']:.3f}")
    min_h = min(left.shape[0], middle.shape[0], right.shape[0])
    panel = np.hstack([left[:min_h], middle[:min_h], right[:min_h]])
    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, panel)


def make_m1m2_panel(img0, cam01, mask, m1, m2, method_name, layer_name, out_path):
    overlay = overlay_cam(img0, cam01, ALPHA)
    mask_bgr = mask_to_bgr(mask)
    panels = [("Original", img0), ("CAM", overlay), ("Otsu Mask", mask_bgr), ("M1 removed", m1), ("M2 kept", m2)]
    titled = [add_title(im, title) for title, im in panels]
    min_h = min(im.shape[0] for im in titled)
    panel = np.hstack([im[:min_h] for im in titled])
    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, panel)


# ============================================================
# CSV WRITERS
# ============================================================

def write_csv(rows, path, fieldnames=None):
    ensure_dir(os.path.dirname(path))
    if not rows:
        return
    if fieldnames is None:
        fieldnames = []
        for r in rows:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_detection_csv_and_summary(rows):
    write_csv(rows, DETECTION_IOU_CSV_PATH)
    valid_best = [float(r["best_iou"]) for r in rows if r["best_iou"] != ""]
    valid_mean_gt = [float(r["mean_best_gt_iou"]) for r in rows if r["mean_best_gt_iou"] != ""]
    total = len(rows)
    images_with_gt = sum(int(r["gt_count"]) > 0 for r in rows)
    images_with_pred = sum(int(r["pred_count"]) > 0 for r in rows)
    tp50 = sum(int(r["tp50"]) for r in rows)
    tp75 = sum(int(r["tp75"]) for r in rows)
    summary = {"total_images": total, "images_with_gt": images_with_gt, "images_with_pred": images_with_pred, "mean_best_iou": float(np.mean(valid_best)) if valid_best else 0.0, "median_best_iou": float(np.median(valid_best)) if valid_best else 0.0, "mean_best_gt_iou": float(np.mean(valid_mean_gt)) if valid_mean_gt else 0.0, "tp50_count": tp50, "tp75_count": tp75, "detection_rate_50": tp50/images_with_gt if images_with_gt else 0.0, "detection_rate_75": tp75/images_with_gt if images_with_gt else 0.0, "conf_thres": CONF_THRES, "nms_iou_thres": NMS_IOU_THRES, "img_size": IMG_SIZE}
    write_csv([summary], DETECTION_IOU_SUMMARY_CSV_PATH)
    print("\nDETECTION IOU SUMMARY")
    for k, v in summary.items():
        print(f"{k:25s}: {v}")


def write_cam_csv_and_sizewise_summary(rows):
    write_csv(rows, CAM_OBJECT_IOU_CSV_PATH)
    grouped = {}
    for r in rows:
        key = (r["method"], r["layer"], r["threshold_type"], float(r["threshold"]), r["size_class"])
        grouped.setdefault(key, []).append(r)
    summary_rows = []
    for key, group in grouped.items():
        method, layer, th_type, threshold, size_class = key
        valid_ious = [float(r["cam_iou"]) for r in group if r["cam_iou"] != ""]
        total_objects = len(group)
        hit50 = sum(int(r["hit50"]) for r in group)
        hit75 = sum(int(r["hit75"]) for r in group)
        summary_rows.append({"method": method, "layer": layer, "threshold_type": th_type, "threshold": threshold, "size_class": size_class, "total_objects": total_objects, "valid_cam_count": len(valid_ious), "mean_cam_iou": float(np.mean(valid_ious)) if valid_ious else 0.0, "median_cam_iou": float(np.median(valid_ious)) if valid_ious else 0.0, "max_cam_iou": float(np.max(valid_ious)) if valid_ious else 0.0, "hit50_count": hit50, "hit75_count": hit75, "hit50_rate": hit50/total_objects if total_objects else 0.0, "hit75_rate": hit75/total_objects if total_objects else 0.0})
    summary_rows = sorted(summary_rows, key=lambda x: (x["method"], x["threshold_type"], x["threshold"], x["size_class"], -x["mean_cam_iou"]))
    write_csv(summary_rows, CAM_SIZE_SUMMARY_CSV_PATH)
    top = sorted(summary_rows, key=lambda x: x["mean_cam_iou"], reverse=True)[:25]
    write_csv(top, os.path.join(OUT_DIR, "csv", "cam_iou_top25.csv"))
    print("\nCAM-IoU CSV saved:", CAM_OBJECT_IOU_CSV_PATH)
    print("CAM-IoU summary saved:", CAM_SIZE_SUMMARY_CSV_PATH)


# ============================================================
# M1/M2 VAL HELPERS
# ============================================================

def make_data_yaml(dataset_root, yaml_path):
    cfg = {"path": os.path.abspath(dataset_root), "train": "images", "val": "images", "test": "images", "nc": NC, "names": CLASS_NAMES}
    ensure_dir(os.path.dirname(yaml_path))
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def run_val(data_yaml, tag):
    cmd_new = [sys.executable, VAL_SCRIPT, "--weights", WEIGHTS, "--data", data_yaml, "--imgsz", str(IMG_SIZE), "--batch-size", str(BATCH_SIZE), "--device", DEVICE_ID]
    cmd_old = [sys.executable, VAL_SCRIPT, "--weights", WEIGHTS, "--data", data_yaml, "--img", str(IMG_SIZE), "--batch", str(BATCH_SIZE), "--device", DEVICE_ID]
    for cmd in [cmd_new, cmd_old]:
        try:
            result = subprocess.run(cmd, cwd=YOLO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="ignore", check=True)
            out = ANSI_ESCAPE.sub("", result.stdout)
            log_dir = os.path.join(OUT_DIR, "logs")
            ensure_dir(log_dir)
            with open(os.path.join(log_dir, f"eval_{safe_name(tag)}.txt"), "w", encoding="utf-8") as f:
                f.write(out)
            for line in out.splitlines():
                if line.strip().startswith("all"):
                    parts = line.split()
                    return {"tag": tag, "P": float(parts[-4]), "R": float(parts[-3]), "mAP50": float(parts[-2]), "mAP50-95": float(parts[-1])}
        except subprocess.CalledProcessError as e:
            if e.stdout and "unrecognized arguments" in e.stdout:
                continue
            raise
    return None


def copy_labels_for_dataset(dst_root, image_paths):
    label_dst = os.path.join(dst_root, "labels")
    ensure_dir(label_dst)
    for p in image_paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        src = os.path.join(LABEL_DIR, stem + ".txt")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(label_dst, stem + ".txt"))


def prepare_baseline_dataset(img_paths):
    root = os.path.join(OUT_DIR, "m1m2_datasets", "baseline")
    if os.path.exists(root):
        shutil.rmtree(root)
    img_dst = os.path.join(root, "images")
    ensure_dir(img_dst)
    for p in img_paths:
        shutil.copy(p, os.path.join(img_dst, os.path.basename(p)))
    copy_labels_for_dataset(root, img_paths)
    yml = os.path.join(OUT_DIR, "m1m2_datasets", "baseline.yaml")
    make_data_yaml(root, yml)
    return root, yml


# ============================================================
# MAIN ANALYSIS
# ============================================================

def run_cam_iou_analysis(model, net, method_layers, imgs):
    detection_rows = []
    cam_object_rows = []
    for idx, pth in enumerate(imgs, 1):
        image_name = os.path.splitext(os.path.basename(pth))[0]
        try:
            x, img0, (h0, w0) = load_image_no_pad(pth)
            label_path = os.path.join(LABEL_DIR, image_name + ".txt")
            gt_boxes, gt_classes = load_yolo_labels(label_path, w0, h0)
            det_640 = get_detections(model, x)
            det_orig = scale_det_to_original(det_640, w0, h0)
            det_row = evaluate_detection_iou(image_name, det_orig, gt_boxes, gt_classes)
            detection_rows.append(det_row)
            if PRINT_EVERY_IMAGE_IOU:
                best_iou = det_row["best_iou"]
                best_iou_txt = "NA" if best_iou == "" else f"{best_iou:.4f}"
                print(f"[DET] {image_name} | GT={det_row['gt_count']} | Pred={det_row['pred_count']} | BestIoU={best_iou_txt}")
            for method_name in METHODS:
                for layer_info in method_layers[method_name]:
                    layer_name = layer_info["name"]
                    layer_module = layer_info["module"]
                    try:
                        cam01 = compute_cam_for_layer(method_name, model, layer_module, x, det_640, w0, h0)
                        otsu_m, otsu_info = otsu_mask(cam01)
                        rows = compute_cam_object_iou_rows(image_name, method_name, layer_name, cam01, gt_boxes, gt_classes, w0, h0, otsu_info)
                        cam_object_rows.extend(rows)
                        if RUN_SAVE_VISUALS:
                            if SAVE_OVERLAY_ONLY:
                                overlay_dir = os.path.join(OUT_DIR, "overlay", method_name, layer_name)
                                ensure_dir(overlay_dir)
                                cv2.imwrite(os.path.join(overlay_dir, f"{image_name}.png"), overlay_cam(img0, cam01, ALPHA))
                            if SAVE_OTSU_MASK:
                                mask_dir = os.path.join(OUT_DIR, "otsu_mask", method_name, layer_name)
                                ensure_dir(mask_dir)
                                cv2.imwrite(os.path.join(mask_dir, f"{image_name}.png"), mask_to_bgr(otsu_m))
                            if SAVE_PANEL:
                                panel_dir = os.path.join(OUT_DIR, "panel", method_name, layer_name)
                                ensure_dir(panel_dir)
                                make_visual_panel(img0, cam01, method_name, layer_name, det_orig, gt_boxes, rows, os.path.join(panel_dir, f"{image_name}.png"))
                    except Exception as e:
                        print(f"[CAM-ERROR] image={image_name} | method={method_name} | layer={layer_name} | error={e}")
                    finally:
                        cleanup_cuda(False)
            if idx % PRINT_PROGRESS_EVERY == 0:
                print(f"[PROGRESS] {idx}/{len(imgs)} images processed")
            gc.collect()
        except Exception as e:
            print(f"[IMAGE-ERROR] {image_name}: {e}")
    write_detection_csv_and_summary(detection_rows)
    write_cam_csv_and_sizewise_summary(cam_object_rows)


def run_m1m2_mapdrop(model, net, m1m2_layers, imgs):
    print("\n" + "="*90)
    print("M1/M2 MAP-DROP START")
    print("="*90)
    results = []
    baseline_root, baseline_yaml = prepare_baseline_dataset(imgs)
    r = run_val(baseline_yaml, "baseline")
    if r:
        results.append(r)
    for method_name in METHODS:
        for layer_info in m1m2_layers[method_name]:
            layer_name = layer_info["name"]
            layer_module = layer_info["module"]
            exp_name = f"{method_name}_{layer_name}"
            exp_safe = safe_name(exp_name)
            m1_root = os.path.join(OUT_DIR, "m1m2_datasets", exp_safe + "_M1")
            m2_root = os.path.join(OUT_DIR, "m1m2_datasets", exp_safe + "_M2")
            for root in [m1_root, m2_root]:
                if os.path.exists(root):
                    shutil.rmtree(root)
                ensure_dir(os.path.join(root, "images"))
                ensure_dir(os.path.join(root, "labels"))
            print(f"\n[M1M2] {exp_name}")
            for idx, pth in enumerate(imgs, 1):
                image_name = os.path.splitext(os.path.basename(pth))[0]
                try:
                    x, img0, (h0, w0) = load_image_no_pad(pth)
                    det_640 = get_detections(model, x)
                    cam01 = compute_cam_for_layer(method_name, model, layer_module, x, det_640, w0, h0)
                    mask, info = otsu_mask(cam01)
                    m1, m2 = make_m1_m2(img0, mask)
                    cv2.imwrite(os.path.join(m1_root, "images", os.path.basename(pth)), m1)
                    cv2.imwrite(os.path.join(m2_root, "images", os.path.basename(pth)), m2)
                    if SAVE_M1M2_SAMPLE_PANELS and idx <= 8:
                        panel_dir = os.path.join(OUT_DIR, "m1m2_panel", exp_safe)
                        ensure_dir(panel_dir)
                        make_m1m2_panel(img0, cam01, mask, m1, m2, method_name, layer_name, os.path.join(panel_dir, f"{image_name}.png"))
                    if idx % 50 == 0:
                        print(f"[M1M2] {idx}/{len(imgs)}")
                    cleanup_cuda(False)
                except Exception as e:
                    print(f"[M1M2-IMAGE-ERROR] {image_name} | {exp_name} | {e}")
            copy_labels_for_dataset(m1_root, imgs)
            copy_labels_for_dataset(m2_root, imgs)
            yml1 = os.path.join(OUT_DIR, "m1m2_datasets", exp_safe + "_M1.yaml")
            yml2 = os.path.join(OUT_DIR, "m1m2_datasets", exp_safe + "_M2.yaml")
            make_data_yaml(m1_root, yml1)
            make_data_yaml(m2_root, yml2)
            r1 = run_val(yml1, exp_safe + "_M1")
            r2 = run_val(yml2, exp_safe + "_M2")
            if r1:
                results.append(r1)
            if r2:
                results.append(r2)
            write_m1m2_results(results)
            cleanup_cuda(True)
    write_m1m2_results(results)
    print("M1/M2 MAP-DROP DONE")


def write_m1m2_results(results):
    baseline = next((r for r in results if r.get("tag") == "baseline"), None)
    out_rows = []
    for r in results:
        rr = dict(r)
        if baseline and rr["tag"] != "baseline":
            rr["mAP50_drop"] = baseline["mAP50"] - rr["mAP50"]
            rr["mAP50_retention"] = rr["mAP50"] / baseline["mAP50"] if baseline["mAP50"] else 0.0
            rr["mAP50-95_drop"] = baseline["mAP50-95"] - rr["mAP50-95"]
            rr["mAP50-95_retention"] = rr["mAP50-95"] / baseline["mAP50-95"] if baseline["mAP50-95"] else 0.0
        out_rows.append(rr)
    write_csv(out_rows, M1M2_MAPDROP_CSV_PATH)
    print("[CSV] M1/M2 saved:", M1M2_MAPDROP_CSV_PATH)


# ============================================================
# RUN
# ============================================================

def run():
    ensure_dir(OUT_DIR)
    ensure_dir(os.path.join(OUT_DIR, "csv"))
    print("="*90)
    print("YOLOv5 SSDD XAI FINAL: CAM-IoU + Otsu + M1/M2 mAP-drop")
    print("="*90)
    print("Device:", DEVICE)
    print("Weights:", WEIGHTS)
    print("OUT_DIR:", OUT_DIR)
    print(f"RUN_CAM_IOU_ANALYSIS={RUN_CAM_IOU_ANALYSIS} | RUN_M1M2_MAPDROP={RUN_M1M2_MAPDROP} | N_IMAGES={N_IMAGES}")

    model = DetectMultiBackend(WEIGHTS, device=DEVICE)
    net = disable_inplace(model.model).eval()
    for p in net.parameters():
        p.requires_grad_(True)

    method_layers = build_method_layers(net, METHOD_LAYER_SPECS)
    m1m2_layers = build_method_layers(net, M1M2_LAYER_SPECS)

    print("\n[INFO] CAM-IoU selected method/layers:")
    for method_name, layers in method_layers.items():
        print(f"\n{method_name}:")
        for layer in layers:
            print(" -", layer["name"])

    all_imgs = []
    for ext in IMAGE_EXTS:
        all_imgs.extend(glob.glob(os.path.join(IMG_DIR, f"*{ext}")))
    all_imgs = sorted(all_imgs)
    imgs = all_imgs if N_IMAGES is None else all_imgs[:N_IMAGES]
    print(f"\n[INFO] Total images selected: {len(imgs)}")

    if RUN_CAM_IOU_ANALYSIS:
        run_cam_iou_analysis(model, net, method_layers, imgs)

    if RUN_M1M2_MAPDROP:
        # M1/M2 mutlaka tüm test görüntüsüyle yapılmalı. N_IMAGES testse sadece deneme olur.
        run_m1m2_mapdrop(model, net, m1m2_layers, imgs)

    print("\nDONE. Results saved to:", OUT_DIR)


if __name__ == "__main__":
    run()
