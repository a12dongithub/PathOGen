import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading ResNet50 model...")
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity() # Remove the final classification layer to get 2048-d features
    model.eval().to(device)
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    parquet_path = r"data/processed/classification/tcga_subtypes/embeddings/combined/benchmark_features.parquet"
    print(f"Loading existing features from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    image_paths = df['image_path'].values
    resnet_features = []
    
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Extracting ResNet50 Features"):
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.amp.autocast('cuda'):
                out = model(tensor)
            resnet_features.append(out[0].cpu().numpy())
            
    df['resnet50'] = list(resnet_features)
    
    print("\nSaving updated features to parquet...")
    df.to_parquet(parquet_path)
    print("Saved to benchmark_features.parquet successfully!")

if __name__ == "__main__":
    main()
