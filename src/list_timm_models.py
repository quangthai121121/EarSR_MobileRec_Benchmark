from __future__ import annotations
import argparse
import timm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern', type=str, default='*')
    parser.add_argument('--pretrained', action='store_true')
    args = parser.parse_args()
    models = timm.list_models(args.pattern, pretrained=args.pretrained)
    for m in models:
        print(m)
    print(f'\nTotal: {len(models)}')


if __name__ == '__main__':
    main()
