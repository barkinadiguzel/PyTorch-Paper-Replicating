# 🖼️ Vision Transformer (ViT) for Image Classification  

Beginner-friendly project demonstrating how to train and use a Vision Transformer (ViT) model for classifying images of 🍕 pizza, 🥩 steak, and 🍣 sushi. Initially trained from scratch, then improved using a pretrained ViT-Base model.  

## 📖 Description  

In this project, I first implemented a Vision Transformer (ViT) model from scratch based on the paper ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/pdf/2010.11929). 📝  

I trained this scratch model on a small custom dataset containing images of pizza, steak, and sushi. Due to the limited size of the dataset, the scratch model’s performance was suboptimal ⚠️, demonstrating that ViT requires large-scale data for effective training.  

To overcome this, I switched to a pretrained ViT-Base model from `torchvision.models`, which was already trained on ImageNet-1k 📊. I froze the base parameters to retain the pretrained features and replaced the classifier head to match the dataset’s three classes 🎯. This approach allowed the model to leverage learned representations from a large dataset while adapting to our specific problem.  

## ⚡ Quick Start  

```bash
pip install torch torchvision
```
## Feedback

For feedback or questions, contact: [barkin.adiguzel@gmail.com](mailto:barkin.adiguzel@gmail.com)
