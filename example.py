from dotsocr import DotsOCR, Result
from more_itertools import chunked
import pymupdf
import time
import json

model = DotsOCR(r"C:\Users\xjq70\Desktop\dots.ocr\weights\DotsOCR")
images = [
    pymupdf.utils.get_pixmap(page, dpi=300).pil_image()
    for page in pymupdf.open(r"C:\Users\xjq70\Downloads\2006.16038v1.pdf")
]

result: list[Result] = []
t0 = time.time()
for chunk in chunked(images, 2):
    chunk_result = model.inference(chunk)
    result.extend(chunk_result)
    print(f"Processed {len(result)}/{len(images)} pages, time: {time.time() - t0:.2f}s")

with open("result.json", "w", encoding="utf-8") as f:
    json.dump([r.model_dump() for r in result], f, ensure_ascii=False, indent=4)
