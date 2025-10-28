from transformers import pipeline

pipe = pipeline("image-classification", model="microsoft/resnet-50")
out = pipe("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/parrots.png")
print(out)