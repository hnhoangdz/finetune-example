# Example of how to Fine-tuning a Deep Learning model using Pytorch
## Prerequites

```
python >= 3.8
pip install -r requirements.txt
CUDA >= 11.4
```

## 1. Understand architecture of pretrained-models, some requirements

- Large pretrained-data (Fer2013 + FerPlus + CK)
- Model must be enough complex

## 2. Define specific layers that will be updated, some requirements (see finetune.py)

- Unfreeze all layers except defined specific layers above
- Append specific layers name to list and push to optimizer (Pytorch)
- Specific layers should stay in the last layers

## 3. Finetune

- [Pretrained-data](https://drive.google.com/file/d/1SrJGftqih4MCLjrT2xJgNXOLQAYKXbbB/view?usp=sharing)
- [Finetuning-data](https://drive.google.com/file/d/1Wdojeyb2il-qqprRyC6oHzK4k4X6H0mE/view?usp=sharing)
- Push data into dataset folder
- Run the code in train.py to pretrain
- Run the code in finetune.py to finetune
