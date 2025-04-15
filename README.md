# Paraphrase Detection with Roberta-base

## 📌 Overview

This repository hosts the quantized version of the Roberta-base model for Paraphrase Detection. The model is designed to determine whether two sentences convey the same meaning. If they are similar, the model outputs "duplicate" with a confidence score; otherwise, it outputs "not duplicate" with a confidence score. The model has been optimized for efficient deployment while maintaining reasonable accuracy, making it suitable for real-time applications.

## 🏗 Model Details

- **Model Architecture:** Roberta-base  
- **Task:** Paraphrase Detection  
- **Dataset:** Hugging Face's `quora-question-pairs`  
- **Quantization:** Float16 (FP16) for optimized inference  
- **Fine-tuning Framework:** Hugging Face Transformers  

## 🚀 Usage

### Installation

```bash
pip install transformers torch
```

### Loading the Model

```python
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "AventIQ-AI/roberta-paraphrase-detection"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name).to(device)
```

### Paraphrase Detection Inference

```python
def predict_paraphrase(sentence1, sentence2, threshold=0.96):
    inputs = tokenizer(sentence1, sentence2, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][predicted_class].item()

    label_map = {0: "Not Duplicate", 1: "Duplicate"}

    # Apply a slightly less strict threshold
    if predicted_class == 1 and confidence < threshold:
        return {"sentence1": sentence1, "sentence2": sentence2, "predicted_label": "Not Duplicate", "confidence": confidence}
    else:
        return {"sentence1": sentence1, "sentence2": sentence2, "predicted_label": label_map[predicted_class], "confidence": confidence}

# 🔍 Test Example
test_cases = [
    ("The sun rises in the east.", "The east is where the sun rises."),  # Duplicate
    ("She enjoys playing the piano.", "She loves playing musical instruments."),  # Duplicate
    ("I had a great time at the party.", "The event was really fun."),  # Duplicate

    ("The sky is blue.", "Bananas are yellow."),  # Not Duplicate
    ("The capital of France is Paris.", "Berlin is the capital of Germany."),  # Not Duplicate
    ("I like reading books.", "She is going for a run."),  # Not Duplicate
]
for sent1, sent2 in test_cases:
    result = predict_paraphrase(sent1, sent2)
    print(result)
```

## 📊 Quantized Model Evaluation Results

### 🔥 Evaluation Metrics 🔥

- ✅ **Accuracy:**  0.7515  
- ✅ **Precision:** 0.6697  
- ✅ **Recall:**    0.5840  
- ✅ **F1-score:**  0.6022  

## ⚡ Quantization Details

Post-training quantization was applied using PyTorch's built-in quantization framework. The model was quantized to Float16 (FP16) to reduce model size and improve inference efficiency while balancing accuracy.

## 📂 Repository Structure

```
.
├── model/               # Contains the quantized model files
├── tokenizer_config/    # Tokenizer configuration and vocabulary files
├── model.safetensors/   # Quantized Model
├── README.md            # Model documentation
```

## ⚠️ Limitations

- The model may struggle with highly nuanced paraphrases.
- Quantization may lead to slight degradation in accuracy compared to full-precision models.
- Performance may vary across different domains and sentence structures.

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request if you have suggestions or improvements.
