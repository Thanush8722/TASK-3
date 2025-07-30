import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

print("✅ Training started with ResNet18")

# Paths
data_dir = "dataset/train"
model_path = "models/resnet_cat_dog.pth"

# ✅ Transforms (use Normalize for pretrained models)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                         [0.229, 0.224, 0.225])  # ImageNet std
])

# ✅ Dataset + Force class_to_idx order
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataset.classes = ['cat', 'dog']
dataset.class_to_idx = {'cat': 0, 'dog': 1}

# ✅ Split into training + validation (80/20)
val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

# ✅ Use pretrained ResNet18
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: cat/dog

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ✅ Class balancing
counts = [0, 0]
for _, lbl in dataset.samples:
    counts[lbl] += 1

weights = torch.tensor([max(counts)/c for c in counts], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 30

# ✅ Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(imgs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(output, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss / total:.4f} - Train Acc: {acc:.2f}%")

    # ✅ Validation accuracy
    model.eval()
    with torch.no_grad():
        val_correct = 0
        val_total = 0
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            _, preds = torch.max(out, 1)
            val_total += labels.size(0)
            val_correct += (preds == labels).sum().item()
        val_acc = 100 * val_correct / val_total
        print(f"➡️  Validation Accuracy: {val_acc:.2f}%")

# ✅ Save model + label mapping
os.makedirs(os.path.dirname(model_path), exist_ok=True)
torch.save({
    "model_state": model.state_dict(),
    "class_to_idx": dataset.class_to_idx
}, model_path)

print("✅ ResNet18 model saved:", model_path)
