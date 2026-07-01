import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pillow_heif import register_heif_opener
from glob import glob
from sklearn.model_selection import StratifiedKFold
import numpy as np
import copy

register_heif_opener()

class ScreenDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, idx):
        img_path = self.paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (256, 256), color='black')
        
        if self.transform:
            img = self.transform(img)
            
        return img, self.labels[idx]

def train_model(model, dataloaders, criterion, optimizer, num_epochs=10):
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
    model.load_state_dict(best_model_wts)
    return model, best_acc

def main():
    real_paths = glob("Dataset/Real/*.*")
    screen_paths = glob("Dataset/Screen/*.*")
    
    all_paths = np.array(real_paths + screen_paths)
    all_labels = np.array([0]*len(real_paths) + [1]*len(screen_paths))
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_accs = []
    
    # Just run 1 fold to see how good it is to save time, or 5 if it's fast
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_paths, all_labels)):
        print(f"Fold {fold+1}")
        
        train_ds = ScreenDataset(all_paths[train_idx], all_labels[train_idx], transform=train_transform)
        val_ds = ScreenDataset(all_paths[val_idx], all_labels[val_idx], transform=val_transform)
        
        dataloaders = {
            'train': DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0),
            'val': DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
        }
        
        # SqueezeNet is EXTREMELY small and fast, MobileNetV2 is also small
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.last_channel, 2)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        model, best_acc = train_model(model, dataloaders, criterion, optimizer, num_epochs=15)
        print(f"Fold {fold+1} Acc: {best_acc:.4f}")
        fold_accs.append(best_acc.item())
        
        # If we got one good model, let's just save it and break so we don't take too long
        torch.save(model.state_dict(), "best_model.pth")
        break
        
    print(f"Estimated Accuracy from Fold 1: {fold_accs[0]:.4f}")

if __name__ == '__main__':
    main()
