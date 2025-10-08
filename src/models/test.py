from transformers import AutoImageProcessor, IJepaModel
import torch
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

image_processor = AutoImageProcessor.from_pretrained("facebook/sparsh-ijepa-base")
model = IJepaModel.from_pretrained("facebook/sparsh-ijepa-base")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    print(outputs.keys())
    print(outputs.last_hidden_state.shape)  # (batch_size, seq_len, hidden_size)
    exit()
    # logits = model(**inputs).logits
    # print(logits.shape)
    # exit()

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])