import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

train_dir = r'C:\Users\athit\OneDrive\Documents\Chest X-Ray ML Project\chest_xray\train'
val_dir = r'C:\Users\athit\OneDrive\Documents\Chest X-Ray ML Project\chest_xray\val'
test_dir = r'C:\Users\athit\OneDrive\Documents\Chest X-Ray ML Project\chest_xray\test'

print("Class mapping:", datasets.ImageFolder(train_dir).class_to_idx)
print("Training set:")
print("  Pneumonia:", len(os.listdir(os.path.join(train_dir, "PNEUMONIA"))))
print("  Normal:", len(os.listdir(os.path.join(train_dir, "NORMAL"))))

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(train_dir, transform=transform_train)
val_data = datasets.ImageFolder(val_dir, transform=transform_test)
test_data = datasets.ImageFolder(test_dir, transform=transform_test)

train_loader = data.DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = data.DataLoader(val_data, batch_size=8, shuffle=False)
test_loader = data.DataLoader(test_data, batch_size=8, shuffle=False)

class_names = ["Normal", "Pneumonia"]

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model = CNNModel().to(device)

class_counts = [len(os.listdir(os.path.join(train_dir, "NORMAL"))),
                len(os.listdir(os.path.join(train_dir, "PNEUMONIA")))]
class_weights = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct_preds += torch.sum(preds == labels).item()
        total_preds += labels.size(0)

    print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Accuracy: {correct_preds/total_preds:.4f}")

model.eval()
correct_preds = 0
total_preds = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        correct_preds += torch.sum(preds == labels).item()
        total_preds += labels.size(0)

test_accuracy = correct_preds / total_preds
print(f"Test Accuracy: {test_accuracy:.4f}")

data_iter = iter(test_loader)
X_new, _ = next(data_iter)
X_new = X_new.to(device)

model.eval()
with torch.no_grad():
    y_proba = torch.softmax(model(X_new), dim=1)
    confidences = y_proba.cpu().numpy().max(axis=1)
    y_pred = torch.argmax(y_proba, dim=1).cpu().numpy()

threshold = 0.6
for i, conf in enumerate(confidences):
    if conf < threshold:
        y_pred[i] = -1

def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).to(tensor.device).view(3, 1, 1)
    std = torch.tensor(std).to(tensor.device).view(3, 1, 1)
    return tensor * std + mean

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for index, ax in enumerate(axes.flatten()):
    image = X_new[index].cpu()
    image = denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ax.imshow(image.permute(1, 2, 0).numpy())
    ax.axis('off')
    
    if y_pred[index] == -1:
        title = "Uncertain Prediction"
    else:
        title = f"Predicted: {class_names[y_pred[index]]}"
    ax.set_title(title, fontsize=12)

plt.subplots_adjust(wspace=0.5)
plt.savefig(r'C:\Users\athit\OneDrive\Documents\Chest X-Ray ML Project\figure1.png')
plt.show()

print(os.listdir(r'C:\Users\athit\OneDrive\Documents\Chest X-Ray ML Project\chest_xray\val\PNEUMONIA'))
