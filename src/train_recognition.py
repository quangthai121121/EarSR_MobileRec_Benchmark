from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import timm

from .common import (
    load_config,
    set_seed,
    seed_worker,
    ensure_dir,
    resolve_device,
    list_images,
    rel_to_root,
)

IMAGENET_MEAN_PAD = (124, 116, 104)

# ---------------------------------------------------------------------------
# LETTERBOX PREPROCESSING (recognition input)
# ---------------------------------------------------------------------------
# Mục đích: đưa ảnh tai (thường không vuông) vào backbone mobile 224×224 mà
# không làm méo giải phẫu và không cắt mất vùng tai.
#
# Cách làm:
#   1. scale = min(image_size / W, image_size / H)
#   2. resize đều 2 chiều (bicubic) → ảnh vừa khít trong canvas vuông
#   3. pad đối xứng viền trống → tensor cố định image_size × image_size
#   4. ToTensor + Normalize ImageNet (mean/std) vì dùng timm pretrained
#
# Vì sao không Resize((224,224))?
#   - Ép vuông làm thay đổi tỉ lệ helix/lobe → đặc trưng nhận dạng tai bị biến dạng.
#
# Vì sao không CenterCrop?
#   - Tai thường chiếm gần hết khung; crop giữa dễ cắt mất biên tai.
#
# Vì sao SR vẫn có thể giúp dù output cuối vẫn 224×224?
#   - LR sau giảm 10% mỗi chiều (vd. 480×560 → 432×504) ít upscale hơn LR giảm 15%;
#     SR ×4 tạo ảnh trung gian sắc hơn trước letterbox 224×224.
#   - Lợi ích là chất lượng nội dung trong 224×224, không phải số pixel đưa vào model.
#
# Công bằng giữa pipeline:
#   - lr_{tag}, bicubic, và mọi pipeline SR dùng CÙNG letterbox + normalize.
#
# Gợi ý diễn giải trong báo cáo (tiếng Anh):
#   "We apply aspect-ratio-preserving letterbox resizing with symmetric padding to
#    a fixed 224×224 input, avoiding geometric distortion and spatial cropping that
#    could remove ear structures. Padding uses RGB values approximating the ImageNet
#    mean to reduce boundary artifacts for pretrained backbones. The same preprocessing
#    is applied across all image pipelines for fair comparison."
#
# Gợi ý diễn giải trong báo cáo (tiếng Việt):
#   "Ảnh được resize giữ nguyên tỉ lệ khung hình (letterbox) và pad viền đối xứng
#    về 224×224, tránh méo hình và cắt mất vùng tai. Màu pad gần mean ImageNet để
#    phù hợp backbone pretrained. Cùng một tiền xử lý được áp dụng cho mọi pipeline
#    (baseline downsample và ảnh sau SR) để so sánh công bằng."
# ---------------------------------------------------------------------------


class Letterbox:
    """Fit image inside size×size, preserve aspect ratio, symmetric pad (no crop, no stretch)."""

    def __init__(self, size: int, fill: tuple[int, int, int] = IMAGENET_MEAN_PAD):
        self.size = size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert('RGB')
        w, h = img.size
        scale = min(self.size / w, self.size / h)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        img = img.resize((nw, nh), Image.Resampling.BICUBIC)
        canvas = Image.new('RGB', (self.size, self.size), self.fill)
        canvas.paste(img, ((self.size - nw) // 2, (self.size - nh) // 2))
        return canvas


def parse_letterbox_fill(value: Sequence[int] | None) -> tuple[int, int, int]:
    if value is None:
        return IMAGENET_MEAN_PAD
    if len(value) != 3:
        raise ValueError('letterbox_fill phải là [R, G, B] với 3 số nguyên')
    return tuple(int(v) for v in value)


def build_transforms(
    image_size: int,
    letterbox_fill: tuple[int, int, int] | None = None,
):
    fill = letterbox_fill if letterbox_fill is not None else IMAGENET_MEAN_PAD
    letterbox = Letterbox(image_size, fill=fill)
    train_tf = transforms.Compose([
        letterbox,
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        letterbox,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def save_recognition_preview(
    data_root: str | Path,
    output_root: str | Path,
    image_size: int,
    letterbox_fill: tuple[int, int, int] | None = None,
    extensions: Sequence[str] | None = None,
    splits: Sequence[str] = ('train', 'val', 'test'),
) -> int:
    """Lưu ảnh sau letterbox (trước normalize) để kiểm tra input recognition."""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

    data_root = Path(data_root)
    output_root = ensure_dir(output_root)
    letterbox = Letterbox(image_size, fill=letterbox_fill or IMAGENET_MEAN_PAD)
    count = 0

    for split in splits:
        split_root = data_root / split
        if not split_root.exists():
            continue
        for img_path in tqdm(list_images(split_root, extensions), desc=f'Preview {split_root.name}', leave=False):
            rel = rel_to_root(img_path, split_root)
            out_path = output_root / split / rel
            ensure_dir(out_path.parent)
            with Image.open(img_path) as im:
                letterbox(im).save(out_path)
            count += 1

    return count


def build_loaders(
    data_root: str | Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int = 42,
    letterbox_fill: tuple[int, int, int] | None = None,
):
    data_root = Path(data_root)
    train_tf, eval_tf = build_transforms(image_size, letterbox_fill=letterbox_fill)
    ds_train = datasets.ImageFolder(data_root / 'train', transform=train_tf)
    ds_val = datasets.ImageFolder(data_root / 'val', transform=eval_tf)
    ds_test = datasets.ImageFolder(data_root / 'test', transform=eval_tf)

    generator = torch.Generator()
    generator.manual_seed(seed)

    loaders = {
        'train': DataLoader(
            ds_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=generator,
        ),
        'val': DataLoader(
            ds_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=generator,
        ),
        'test': DataLoader(
            ds_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=generator,
        ),
    }
    return loaders, ds_train.classes


def make_model(model_name: str, num_classes: int, pretrained: bool = True):
    try:
        return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    except Exception as e:
        if pretrained:
            print(f'WARNING: pretrained=True lỗi với {model_name}: {e}')
            print('Fallback sang pretrained=False')
            return timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        raise


def make_optimizer(model, name: str, lr: float, weight_decay: float):
    name = name.lower()
    if name == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f'Unknown optimizer: {name}')


@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, float | np.ndarray]:
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        pred = torch.argmax(logits, dim=1)
        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())
        total_loss += loss.item() * x.size(0)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    return {
        'loss': total_loss / max(1, len(loader.dataset)),
        'accuracy': acc,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1,
        'confusion_matrix': cm,
    }


def train_one(
    data_root: str | Path,
    model_name: str,
    output_dir: str | Path,
    image_size: int = 224,
    batch_size: int = 32,
    epochs: int = 50,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    optimizer_name: str = 'adamw',
    patience: int = 10,
    pretrained: bool = True,
    amp: bool = True,
    num_workers: int = 4,
    seed: int = 42,
    device_name: str = 'cuda',
    letterbox_fill: tuple[int, int, int] | None = None,
) -> Dict[str, float]:
    set_seed(seed, deterministic=True)
    device = resolve_device(device_name)
    output_dir = ensure_dir(output_dir)

    loaders, classes = build_loaders(
        data_root,
        image_size,
        batch_size,
        num_workers,
        seed=seed,
        letterbox_fill=letterbox_fill,
    )
    model = make_model(model_name, num_classes=len(classes), pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, optimizer_name, lr, weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == 'cuda'))

    best_f1 = -1.0
    best_epoch = -1
    no_improve = 0
    history = []
    best_path = output_dir / 'best_model.pt'
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in tqdm(loaders['train'], desc=f'{model_name} epoch {epoch}/{epochs}', leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(amp and device.type == 'cuda')):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * x.size(0)

        val_metrics = evaluate(model, loaders['val'], device)
        train_loss /= max(1, len(loaders['train'].dataset))
        row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': float(val_metrics['loss']),
            'val_accuracy': float(val_metrics['accuracy']),
            'val_precision_macro': float(val_metrics['precision_macro']),
            'val_recall_macro': float(val_metrics['recall_macro']),
            'val_f1_macro': float(val_metrics['f1_macro']),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / 'history.csv', index=False)
        print(row)

        if row['val_f1_macro'] > best_f1:
            best_f1 = row['val_f1_macro']
            best_epoch = epoch
            no_improve = 0
            torch.save({'model': model.state_dict(), 'classes': classes, 'model_name': model_name}, best_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch}. Best epoch={best_epoch}')
                break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    test_metrics = evaluate(model, loaders['test'], device)
    elapsed = time.time() - t0

    result = {
        'model_name': model_name,
        'num_classes': len(classes),
        'best_epoch': best_epoch,
        'test_loss': float(test_metrics['loss']),
        'accuracy': float(test_metrics['accuracy']),
        'precision_macro': float(test_metrics['precision_macro']),
        'recall_macro': float(test_metrics['recall_macro']),
        'f1_macro': float(test_metrics['f1_macro']),
        'elapsed_sec': elapsed,
        'seed': seed,
    }
    pd.DataFrame([result]).to_csv(output_dir / 'metrics.csv', index=False)
    np.savetxt(output_dir / 'confusion_matrix.csv', test_metrics['confusion_matrix'], fmt='%d', delimiter=',')
    with open(output_dir / 'classes.json', 'w', encoding='utf-8') as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)
    print('TEST:', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/benchmark.yaml')
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    tr = cfg['training']
    train_one(
        data_root=args.data_root,
        model_name=args.model_name,
        output_dir=args.output_dir,
        image_size=cfg['data']['image_size'],
        batch_size=args.batch_size or tr['batch_size'],
        epochs=args.epochs or tr['epochs'],
        lr=args.lr or tr['lr'],
        weight_decay=tr['weight_decay'],
        optimizer_name=tr['optimizer'],
        patience=tr['patience'],
        pretrained=tr['pretrained'],
        amp=tr['amp'],
        num_workers=cfg['project']['num_workers'],
        seed=cfg['project']['seed'],
        device_name=cfg['project']['device'],
        letterbox_fill=parse_letterbox_fill(cfg['data'].get('letterbox_fill')),
    )


if __name__ == '__main__':
    main()
