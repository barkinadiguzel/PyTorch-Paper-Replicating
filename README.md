# Vision Transformer (ViT) Tutorial

In this project, I trained a Vision Transformer (ViT) model on a small custom dataset of pizza, steak, and sushi images. Initially, I tried training the model from scratch, but due to the limited size of our dataset, the results were suboptimal. To improve performance, I switched to a pretrained ViT-Base model from torchvision.models, which was already trained on ImageNet-1k. I froze the base parameters to retain the pretrained features and replaced the classifier head to match our dataset’s three classes. This approach allowed the model to leverage learned representations from a large dataset while adapting to our specific problem.

## Quick Start

```bash
pip install torch torchvision
```
## Feedback

For feedback or questions, contact: [barkin.adiguzel@gmail.com](mailto:barkin.adiguzel@gmail.com)
