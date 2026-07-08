from __future__ import annotations
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from .common import ensure_dir, list_images, rel_to_root


def bicubic_upscale_dataset(input_root: str | Path, output_root: str | Path, scale: int = 4, extensions=None) -> int:
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    input_root = Path(input_root)
    output_root = ensure_dir(output_root)
    images = list_images(input_root, extensions)
    for img_path in tqdm(images, desc=f'Bicubic x{scale}'):
        rel = rel_to_root(img_path, input_root)
        out_path = output_root / rel
        ensure_dir(out_path.parent)
        with Image.open(img_path) as im:
            im = im.convert('RGB')
            w, h = im.size
            sr = im.resize((w * scale, h * scale), Image.Resampling.BICUBIC)
            sr.save(out_path)
    return len(images)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', required=True)
    parser.add_argument('--output_root', required=True)
    parser.add_argument('--scale', type=int, default=4)
    args = parser.parse_args()
    n = bicubic_upscale_dataset(args.input_root, args.output_root, args.scale)
    print(f'Done. Processed {n} images.')


if __name__ == '__main__':
    main()
