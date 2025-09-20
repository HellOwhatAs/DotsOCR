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
uv add git+https://github.com/HellOwhatAs/DotsOCR
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

## Prompt
```py
from dotsocr import PROMPT
print(PROMPT)
```
```
Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
```