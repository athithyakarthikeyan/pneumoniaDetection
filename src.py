import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms, models
import numpy as np
import os
from tkinter import filedialog, Tk, Button, Label
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to dataset
train_dir = r"C:\Users\athit\OneDrive\Documents\Chest X-Ray ML Project\chest_xray\train"
test_dir = r"C:\Users\athit\OneDrive\Documents\Chest X-Ray ML Project\chest_xray\test"

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_data = datasets.ImageFolder(train_dir, transform=transform)
test_data = datasets.ImageFolder(test_dir, transform=transform)

train_loader = data.DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = data.DataLoader(test_data, batch_size=8, shuffle=False)

class_names = ["Normal", "Pneumonia"]

# Load pre-trained ResNet50 and modify final layer
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Train model
epochs = 5
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

# Save trained model
torch.save(model.state_dict(), "pneumonia_model.pth")

# Sample images
sample_normal_img = os.path.join(test_dir, "NORMAL", os.listdir(os.path.join(test_dir, "NORMAL"))[0])
sample_pneumonia_img = os.path.join(test_dir, "PNEUMONIA", os.listdir(os.path.join(test_dir, "PNEUMONIA"))[0])

# Grad-CAM Function
def apply_grad_cam(image_path, predicted_class):
    if predicted_class == 0:
        return None  # No need for Grad-CAM if prediction is normal

    model.eval()

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Target last convolutional layer
    target_layer = model.layer4[-1].conv3
    forward_hook_handle = target_layer.register_forward_hook(forward_hook)
    backward_hook_handle = target_layer.register_full_backward_hook(backward_hook)  # Fixed warning

    # Forward pass
    output = model(img_tensor)
    class_idx = torch.argmax(output, dim=1).item()  # Ensure correct class is used

    model.zero_grad()
    output[:, class_idx].backward()

    # Remove hooks
    forward_hook_handle.remove()
    backward_hook_handle.remove()

    # Extract activations & gradients
    activations = activations[0].detach()
    gradients = gradients[0].detach()

    # Compute Grad-CAM heatmap
    pooled_grads = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_grads[i]

    heatmap = torch.mean(activations, dim=1).squeeze().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)  # ReLU to remove negatives
    heatmap /= np.max(heatmap) if np.max(heatmap) > 0 else 1  # Normalize

    # Resize heatmap and apply color mapping
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay on original image
    original_img = np.array(image.resize((224, 224)))  # Convert PIL to NumPy
    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    return superimposed_img

# Prediction function
def predict_image(image_path):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted_class = torch.max(output, 1)
        predicted_class = predicted_class.item()

    prediction = class_names[predicted_class]
    heatmap_img = apply_grad_cam(image_path, predicted_class)

    if predicted_class == 0:
        display_images(image_path, sample_normal_img, sample_pneumonia_img, None, prediction)
    else:
        heatmap_path = "gradcam_result.jpg"
        cv2.imwrite(heatmap_path, cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2BGR))
        display_images(image_path, sample_normal_img, sample_pneumonia_img, heatmap_path, prediction)

# Display images in Matplotlib
def display_images(user_img, normal_img, pneumonia_img, heatmap_img, prediction):
    images = [user_img, normal_img, pneumonia_img]
    titles = ["Your X-ray", "Normal Lung", "Pneumonia Lung"]
    
    images_to_show = []
    for img_path in images:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images_to_show.append(img)

    if heatmap_img:
        heatmap = cv2.imread(heatmap_img)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        images_to_show.append(heatmap)
        titles.append("Affected Areas")

    fig, axes = plt.subplots(1, len(images_to_show), figsize=(20, 5))
    for ax, img, title in zip(axes, images_to_show, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    fig.suptitle(f"Prediction: {prediction}", fontsize=15, color="red")
    plt.show()

# GUI for Uploading Images
def upload_and_predict():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        predict_image(file_path)

# Tkinter GUI
root = Tk()
root.title("Pneumonia Detection System")
root.geometry("400x200")
Label(root, text="Pneumonia Detection using Chest X-Ray", font=("Helvetica", 14)).pack(pady=20)
Button(root, text="Upload Chest X-Ray", command=upload_and_predict, font=("Helvetica", 12)).pack(pady=20)
Button(root, text="Exit", command=root.quit, font=("Helvetica", 12)).pack(pady=20)
root.mainloop()
