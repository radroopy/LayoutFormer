#!/usr/bin/env python3
import torch

from ly import FourierFeatureEncoder


def main():
    torch.set_printoptions(precision=6, sci_mode=False)

    encoder = FourierFeatureEncoder(num_bands=4)
    x = torch.tensor([[0.6]])

    print("input:", x)
    out = encoder(x)
    print("output shape:", tuple(out.shape))
    print("output:", out)


if __name__ == "__main__":
    main()
