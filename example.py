from dotsocr import DotsOCR, Result, Category
from io import BytesIO
from more_itertools import chunked
import pymupdf
import base64
import time
import json

model_path = r"C:\Users\xjq70\Desktop\dots.ocr\weights\DotsOCR"
pdf_path = r"C:\Users\xjq70\Downloads\2006.16038v1.pdf"

model = DotsOCR(model_path)
images = [
    pymupdf.utils.get_pixmap(page, dpi=300).pil_image()
    for page in pymupdf.open(pdf_path)
]

result: list[Result] = []
t0 = time.time()
for chunk in chunked(images, 2):
    chunk_result = model.inference(chunk)
    result.extend(chunk_result)
    print(f"Processed {len(result)}/{len(images)} pages, time: {time.time() - t0:.2f}s")

with open("result.json", "w", encoding="utf-8") as f:
    json.dump([r.model_dump() for r in result], f, ensure_ascii=False, indent=4)

md_result: list[str] = []
for page, img in zip(result, images):
    for elem in page.root:
        if elem.category in {Category.PageFooter, Category.PageHeader}:
            continue
        elif elem.category == Category.Picture:
            buffer = BytesIO()
            img.crop(elem.bbox.to_list()).save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            buffer.close()
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            md_result.append(f"![](data:image/png;base64,{img_base64})")
        else:
            md_result.append(elem.text)
            if elem.category == Category.Table:
                md_result.append("")

with open("result.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md_result))
