from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from tqdm import tqdm

from .common import ensure_dir, list_images, rel_to_root, reduce_percent_to_keep_ratio

DEFAULT_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']
PIL_INTERPOLATION = {
    'bicubic': Image.Resampling.BICUBIC,
    'bilinear': Image.Resampling.BILINEAR,
    'lanczos': Image.Resampling.LANCZOS,
    'area': Image.Resampling.BOX,
}


def center_crop_to_mod(im: Image.Image, scale: int) -> Image.Image:
    """Center-crop image so width and height are divisible by an integer scale."""
    w, h = im.size
    new_w = (w // scale) * scale
    new_h = (h // scale) * scale
    if new_w <= 0 or new_h <= 0:
        raise ValueError(f'Image too small for scale x{scale}: {w}x{h}')
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    return im.crop((left, top, left + new_w, top + new_h))


def add_gaussian_noise(im: Image.Image, std: float, rng: np.random.Generator) -> Image.Image:
    if std <= 0:
        return im
    arr = np.asarray(im).astype(np.float32)
    noise = rng.normal(loc=0.0, scale=std, size=arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_image(im: Image.Image, out_path: Path, output_format: str, jpeg_quality: int) -> Path:
    ensure_dir(out_path.parent)
    fmt = output_format.lower()
    if fmt in {'jpg', 'jpeg'}:
        final_path = out_path.with_suffix('.jpg')
        im.save(final_path, quality=jpeg_quality, subsampling=0)
        return final_path
    if fmt == 'png':
        final_path = out_path.with_suffix('.png')
        im.save(final_path, compress_level=3)
        return final_path
    raise ValueError('output_format chỉ hỗ trợ png hoặc jpg')


def resolve_lr_size(
    hr_w: int,
    hr_h: int,
    scale: int | None,
    ratio: float | None,
    reduce_percent: float | None = None,
) -> tuple[int, int, str]:
    """Return LR size and human-readable degradation label.

    Modes:
    - scale: --scale 4 → width/height còn 1/4.
    - ratio: --ratio 0.9 hoặc --keep_percent 90 → còn 90% mỗi chiều.
    - reduce_percent: --reduce_percent 10 → giảm 10% → còn 90% mỗi chiều.
    """
    if ratio is not None:
        if not (0 < ratio <= 1):
            raise ValueError('--ratio / keep_percent phải trong (0, 1], ví dụ 0.9 cho còn 90%.')
        keep_pct = ratio * 100
        if reduce_percent is not None:
            label = f'giam_{reduce_percent:g}%_con_{keep_pct:.4g}%'
        else:
            label = f'con_{keep_pct:.4g}%'
        return max(1, int(round(hr_w * ratio))), max(1, int(round(hr_h * ratio))), label

    if scale is None:
        scale = 4
    if scale <= 0:
        raise ValueError('--scale phải > 0')
    return max(1, hr_w // scale), max(1, hr_h // scale), f'x{scale}'


def downsample_one(
    img_path: Path,
    input_root: Path,
    output_lr_root: Path,
    scale: int | None,
    ratio: float | None,
    interpolation: str,
    output_format: str,
    mod_crop: bool,
    output_hr_root: Optional[Path],
    degradation: str,
    blur_radius: float,
    noise_std: float,
    jpeg_quality: int,
    rng: np.random.Generator,
    reduce_percent: float | None = None,
) -> dict:
    rel = rel_to_root(img_path, input_root)
    with Image.open(img_path) as im:
        im = ImageOps.exif_transpose(im).convert('RGB')
        orig_w, orig_h = im.size

        # mod_crop chỉ có ý nghĩa khi dùng scale nguyên như x2/x3/x4.
        if mod_crop and ratio is not None:
            raise ValueError('Không dùng --mod_crop chung với --ratio/--keep_percent/--reduce_percent. Dùng --scale nếu cần crop chia hết.')
        hr = center_crop_to_mod(im, scale or 4) if mod_crop else im
        hr_w, hr_h = hr.size
        lr_w, lr_h, downsample_label = resolve_lr_size(
            hr_w, hr_h, scale=scale, ratio=ratio, reduce_percent=reduce_percent,
        )

        if degradation == 'bicubic':
            lr = hr.resize((lr_w, lr_h), PIL_INTERPOLATION[interpolation])
        elif degradation == 'realistic':
            # Degradation đơn giản, deterministic theo seed:
            # blur nhẹ -> downsample -> noise nhẹ -> JPEG-like save quality nếu output jpg.
            work = hr
            if blur_radius > 0:
                work = work.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            lr = work.resize((lr_w, lr_h), PIL_INTERPOLATION[interpolation])
            lr = add_gaussian_noise(lr, noise_std, rng)
        else:
            raise ValueError(f'Unknown degradation: {degradation}')

        out_lr_path = save_image(lr, output_lr_root / rel, output_format, jpeg_quality)
        out_hr_path = ''
        if output_hr_root is not None:
            out_hr_path = str(save_image(hr, output_hr_root / rel, output_format, jpeg_quality=95))

    return {
        'relative_path': str(rel),
        'lr_path': str(out_lr_path),
        'hr_path': out_hr_path,
        'orig_w': orig_w,
        'orig_h': orig_h,
        'hr_w': hr_w,
        'hr_h': hr_h,
        'lr_w': lr_w,
        'lr_h': lr_h,
        'scale': scale if scale is not None else '',
        'keep_ratio': ratio if ratio is not None else '',
        'reduce_percent': reduce_percent if reduce_percent is not None else '',
        'keep_percent': ratio * 100 if ratio is not None else '',
        'downsample_label': downsample_label,
        'mod_crop': int(mod_crop),
        'degradation': degradation,
    }


def downsample_dataset(
    input_root: str | Path,
    output_lr_root: str | Path,
    scale: int | None = 4,
    ratio: float | None = None,
    interpolation: str = 'bicubic',
    output_format: str = 'png',
    mod_crop: bool = False,
    output_hr_root: str | Path | None = None,
    degradation: str = 'bicubic',
    blur_radius: float = 0.0,
    noise_std: float = 0.0,
    jpeg_quality: int = 95,
    seed: int = 42,
    extensions: Iterable[str] = DEFAULT_EXTENSIONS,
    report_csv: str | Path | None = None,
    reduce_percent: float | None = None,
) -> int:
    input_root = Path(input_root)
    output_lr_root = ensure_dir(output_lr_root)
    output_hr_root_path = ensure_dir(output_hr_root) if output_hr_root else None

    if interpolation not in PIL_INTERPOLATION:
        raise ValueError(f'interpolation phải là một trong: {list(PIL_INTERPOLATION)}')

    images = list_images(input_root, extensions)
    rng = np.random.default_rng(seed)
    random.seed(seed)

    rows = []
    if ratio is not None:
        if reduce_percent is not None:
            label = f'giam {reduce_percent:g}% (con {ratio * 100:.4g}%)'
        else:
            label = f'con {ratio * 100:.4g}%'
    else:
        label = f'x{scale}'
    for img_path in tqdm(images, desc=f'Downsample {label} -> {output_lr_root}'):
        rows.append(downsample_one(
            img_path=img_path,
            input_root=input_root,
            output_lr_root=output_lr_root,
            scale=scale,
            ratio=ratio,
            interpolation=interpolation,
            output_format=output_format,
            mod_crop=mod_crop,
            output_hr_root=output_hr_root_path,
            degradation=degradation,
            blur_radius=blur_radius,
            noise_std=noise_std,
            jpeg_quality=jpeg_quality,
            rng=rng,
            reduce_percent=reduce_percent,
        ))

    if report_csv:
        report_csv = Path(report_csv)
        ensure_dir(report_csv.parent)
        fieldnames = list(rows[0].keys()) if rows else [
            'relative_path', 'lr_path', 'hr_path', 'orig_w', 'orig_h', 'hr_w', 'hr_h',
            'lr_w', 'lr_h', 'scale', 'keep_ratio', 'reduce_percent', 'keep_percent',
            'downsample_label', 'mod_crop', 'degradation'
        ]
        with open(report_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return len(images)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Tạo dataset low-resolution trước khi chạy các model SR. Giữ nguyên cấu trúc train/val/test/class.'
    )
    parser.add_argument('--input_root', required=True, help='VD: data/EarVN1.0_split')
    parser.add_argument('--output_lr_root', required=True, help='VD: outputs/lr/earvn_r10')
    parser.add_argument('--scale', type=int, default=None, help='VD: 4 = width/height còn 1/4. Không dùng chung với --reduce_percent/--keep_percent/--ratio.')
    parser.add_argument('--ratio', type=float, default=None, help='Tỉ lệ GIỮ LẠI mỗi chiều. VD: 0.9 = còn 90%.')
    parser.add_argument(
        '--reduce_percent',
        type=float,
        default=None,
        help='Giảm X%% kích thước mỗi chiều. VD: 10 → còn 90%% (480×560 → 432×504).',
    )
    parser.add_argument(
        '--percent',
        type=float,
        default=None,
        help='Alias của --reduce_percent (cùng nghĩa: giảm X%%).',
    )
    parser.add_argument(
        '--keep_percent',
        type=float,
        default=None,
        help='Còn X%% kích thước mỗi chiều. VD: 90 = giảm 10%%.',
    )
    parser.add_argument('--interpolation', default='bicubic', choices=list(PIL_INTERPOLATION.keys()))
    parser.add_argument('--output_format', default='png', choices=['png', 'jpg', 'jpeg'])
    parser.add_argument('--mod_crop', action='store_true', help='Crop HR để width/height chia hết cho scale. Chỉ dùng với --scale.')
    parser.add_argument('--output_hr_root', default=None, help='Optional: lưu HR sau mod-crop/original để so PSNR/SSIM hoặc làm HR baseline.')
    parser.add_argument('--degradation', default='bicubic', choices=['bicubic', 'realistic'])
    parser.add_argument('--blur_radius', type=float, default=0.0, help='Chỉ dùng khi degradation=realistic')
    parser.add_argument('--noise_std', type=float, default=0.0, help='Gaussian noise std, chỉ dùng khi degradation=realistic')
    parser.add_argument('--jpeg_quality', type=int, default=95)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--report_csv', default='outputs/reports/downsample_report.csv')
    args = parser.parse_args()

    ratio = args.ratio
    reduce_percent = args.reduce_percent

    if args.percent is not None:
        if reduce_percent is not None:
            raise ValueError('Chỉ dùng một trong hai: --reduce_percent hoặc --percent')
        reduce_percent = args.percent

    if args.keep_percent is not None:
        if ratio is not None or reduce_percent is not None:
            raise ValueError('Chỉ dùng một nhóm: --keep_percent HOẶC --reduce_percent/--percent HOẶC --ratio')
        if not (0 < args.keep_percent <= 100):
            raise ValueError('--keep_percent phải trong (0, 100]')
        ratio = args.keep_percent / 100.0
    elif reduce_percent is not None:
        ratio = reduce_percent_to_keep_ratio(reduce_percent)

    if ratio is not None and args.scale is not None:
        raise ValueError('Chỉ dùng một trong hai: --scale hoặc --reduce_percent/--keep_percent/--ratio')
    if ratio is None and args.scale is None:
        args.scale = 4

    n = downsample_dataset(
        input_root=args.input_root,
        output_lr_root=args.output_lr_root,
        scale=args.scale,
        ratio=ratio,
        interpolation=args.interpolation,
        output_format=args.output_format,
        mod_crop=args.mod_crop,
        output_hr_root=args.output_hr_root,
        degradation=args.degradation,
        blur_radius=args.blur_radius,
        noise_std=args.noise_std,
        jpeg_quality=args.jpeg_quality,
        seed=args.seed,
        report_csv=args.report_csv,
        reduce_percent=reduce_percent,
    )
    if reduce_percent is not None:
        keep = 100.0 - reduce_percent
        print(f'Done. Giảm {reduce_percent:g}% mỗi chiều → còn {keep:g}% kích thước width/height.')
    print(f'Done. Created LR dataset with {n} images at: {args.output_lr_root}')
    if args.output_hr_root:
        print(f'HR/mod-crop images saved at: {args.output_hr_root}')
    print(f'Report CSV: {args.report_csv}')


if __name__ == '__main__':
    main()
