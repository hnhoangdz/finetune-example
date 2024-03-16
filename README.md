# Example of how to Fine-tuning a Deep Learning model using Pytorch

## 1. Understand architecture of pretrained-models, some requirements

- Large pretrained-data (Fer2013 + FerPlus + CK)
- Model must be enough complex

## 2. Define specific layers will be update, some requirements (see finetune.py)

- Unfreeze all layers except defined specific layers above
- Append spefic layers name to list and push to optimizer (Pytorch)
- Specific layers should be stay in the last layers

## 3. Finetune
