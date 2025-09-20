from dotsocr import DotsOCR


model = DotsOCR(r"C:\Users\xjq70\Desktop\dots.ocr\weights\DotsOCR")
result = model.inference([r"C:\Users\xjq70\Desktop\llmbook60.png"])
print(result)
