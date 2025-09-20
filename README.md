# dotsocr

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![uv](https://img.shields.io/badge/uv-package_manager-green.svg)](https://github.com/astral-sh/uv)
[![CUDA 12.8](https://img.shields.io/badge/cu128-CUDA%2012.8-brightgreen.svg)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch 2.7.0](https://img.shields.io/badge/pytorch-2.7.0-red.svg)](https://pytorch.org/get-started/locally/)
[![OS Support](https://img.shields.io/badge/support-Windows%20x64%20%7C%20Linux%20x64-blue.svg)](https://github.com/xjq70/dotsocr)

Installable package for [DotsOCR](https://github.com/rednote-hilab/dots.ocr), powered by [uv](https://docs.astral.sh/uv/).

> Supports only Python3.12, Pytorch==2.7.0+cu128 on Windows x64 or Linux x64.

## Install
```
uv add ...
```

## QuickStart
First, download model weights from https://huggingface.co/rednote-hilab/dots.ocr.
```py
from dotsocr import DotsOCR

# load downloaded weights
model = DotsOCR("path/to/DotsOCR")
# batched inference
result = model.inference([
    "image1.png",
    "image2.png",
    # ...
])
print(result)
```