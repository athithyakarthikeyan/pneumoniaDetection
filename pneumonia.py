import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "chest_xray")
MODEL_PATH = 'pneumonia_resnet50_model.pth'
IMG_SIZE = 256
CROP_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomResizedCrop(CROP_SIZE, scale=(0.7, 1.0)),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3))
])

val_test_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(CROP_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
])

class ChestXRayDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.transform = transform
        self.classes = ['NORMAL', 'PNEUMONIA']
        self.image_paths = []
        self.labels = []
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(folder_path, class_name)
            if not os.path.exists(class_dir):
                raise ValueError(f"Missing {class_name} directory in {folder_path}")
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img.verify()
                        self.image_paths.append(img_path)
                        self.labels.append(class_idx)
                    except Exception as e:
                        print(f"Skipping invalid image: {img_path} - {str(e)}")
        
        class_counts = np.bincount(self.labels)
        self.class_weights = 1. / torch.Tensor(class_counts)
        self.samples_weights = self.class_weights[self.labels]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.forward_hook = self.target_layer.register_forward_hook(self.save_activations)
        self.backward_hook = self.target_layer.register_backward_hook(self.save_gradients)
    
    def __del__(self):
        self.forward_hook.remove()
        self.backward_hook.remove()
    
    def save_activations(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
        
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        
        output.backward(gradient=one_hot, retain_graph=True)
        
        if self.activations is None or self.gradients is None:
            raise RuntimeError("GradCAM failed to capture activations/gradients")
        
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.squeeze().cpu().numpy()

def train_model():
    train_dataset = ChestXRayDataset(os.path.join(DATA_DIR, "train"), train_transform)
    val_dataset = ChestXRayDataset(os.path.join(DATA_DIR, "val"), val_test_transform)
    
    print("\nDataset Statistics:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Class distribution (train): {np.bincount(train_dataset.labels)}")
    
    sampler = WeightedRandomSampler(train_dataset.samples_weights, len(train_dataset), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                             pin_memory=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, pin_memory=True)

    model = models.resnet50(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )
    model = model.to(DEVICE)
    
    for layer in model.fc.modules():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0)

    class_weights = train_dataset.class_weights.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_acc = 0.0

    print("\nStarting training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = correct_train / total_train
        val_acc = correct_val / total_val
        
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%}")
        print(f"Val Loss: {val_loss/len(val_loader.dataset):.4f} | Acc: {val_acc:.2%}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"New best model saved at {val_acc:.2%} validation accuracy")

    print(f"\nTraining complete. Best validation accuracy: {best_acc:.2%}")
    return model

def evaluate_model(model):
    test_dataset = ChestXRayDataset(os.path.join(DATA_DIR, "test"), val_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print("\nTest Set Evaluation:")
    print(f"Accuracy: {np.mean(np.array(all_preds) == np.array(all_labels)):.4f}")
    print(f"Precision: {precision_score(all_labels, all_preds):.4f}")
    print(f"Recall: {recall_score(all_labels, all_preds):.4f}")
    print(f"F1 Score: {f1_score(all_labels, all_preds):.4f}")

class PneumoniaApp:
    def __init__(self, root, model):
        self.root = root
        self.root.title("Pneumonia Detection")
        self.root.geometry("1000x800")
        self.model = model
        self.model.eval()
        
        self.gradcam = GradCAM(model, model.layer4[-1])
        self.create_widgets()
    
    def create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(control_frame, text="Select X-Ray", command=self.load_image).pack(side=tk.LEFT, padx=5)
        self.status_label = ttk.Label(control_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        self.figure, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.figure, master=main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.prediction_var = tk.StringVar()
        prediction_frame = ttk.Frame(main_frame)
        prediction_frame.pack(fill=tk.X, pady=5)
        self.prediction_label = ttk.Label(prediction_frame, textvariable=self.prediction_var, 
                                        font=('Arial', 14, 'bold'))
        self.prediction_label.pack()
    
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if not file_path:
            return
        
        self.status_label.config(text="Processing...")
        self.root.update()
        
        try:
            original_image = Image.open(file_path).convert('RGB')
            input_tensor = val_test_transform(original_image).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                prob = torch.softmax(output, dim=1)[0][1].item()
                pred = 1 if prob > 0.5 else 0
            
            self.ax1.clear()
            self.ax2.clear()
            
            self.ax1.imshow(original_image.resize((CROP_SIZE, CROP_SIZE)))
            self.ax1.set_title("Original X-Ray")
            self.ax1.axis('off')

            if pred == 1:
                cam = self.gradcam.generate_cam(input_tensor, target_class=pred)
                heatmap = cv2.resize(cam, (CROP_SIZE, CROP_SIZE))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                
                original_np = np.array(original_image.resize((CROP_SIZE, CROP_SIZE)))
                superimposed = cv2.addWeighted(original_np, 0.6, heatmap, 0.4, 0)
                
                self.ax2.imshow(superimposed)
                self.ax2.set_title("Grad-CAM Heatmap")
                self.ax2.axis('off')
            else:
                self.ax2.set_visible(False)
                self.figure.subplots_adjust(wspace=0)
                self.ax1.set_position([0.1, 0.1, 0.8, 0.8])

            status = "PNEUMONIA" if pred == 1 else "NORMAL"
            confidence = prob if pred == 1 else 1 - prob
            color = "red" if pred == 1 else "green"
            self.prediction_var.set(f"Diagnosis: {status} ({confidence*100:.1f}% confidence)")
            self.prediction_label.config(foreground=color)
            
            self.canvas.draw()
            self.status_label.config(text="Analysis Complete")
            
            self.ax2.set_visible(True)
            self.figure.subplots_adjust(wspace=0.3)

        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")
            self.prediction_var.set("")

if __name__ == "__main__":
    print("Initializing Pneumonia Detection System...")
    
    for split in ["train", "val", "test"]:
        dataset = ChestXRayDataset(os.path.join(DATA_DIR, split))
        print(f"{split.capitalize()}:")
        print(f"  Normal: {sum(1 for l in dataset.labels if l == 0)}")
        print(f"  Pneumonia: {sum(1 for l in dataset.labels if l == 1)}")
    
    if os.path.exists(MODEL_PATH):
        model = models.resnet50()
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model = model.to(DEVICE)
        print("\nLoaded pre-trained model")
    else:
        print("\nTraining new model...")
        model = train_model()
    
    evaluate_model(model)
    
    root = tk.Tk()
    app = PneumoniaApp(root, model)
    root.mainloop()
