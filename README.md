# PyTorch-Paper-Replicating
I trained a ViT model on a small pizza, steak, and sushi dataset. Training from scratch underperformed due to limited data, so I used a pretrained ViT-Base from torchvision, froze its base layers, and updated the classifier head. This transfer learning approach improved accuracy and stabilized training.
