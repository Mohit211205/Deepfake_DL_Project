import torch
from PIL import Image
from torchvision import transforms
from models.deepfake_classifier import DeepfakeClassifier


# Load model
model = DeepfakeClassifier().cuda()
model.load_state_dict(torch.load("outputs/checkpoints/classifier.pth"))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load test image
img_path = input("Enter image path: ")

img = Image.open(img_path).convert("RGB")
img = transform(img).unsqueeze(0).cuda()

# Prediction
with torch.no_grad():
    output = model(img)
    pred = output.argmax(dim=1).item()

if pred == 0:
    print("Prediction: REAL IMAGE")
else:
    print("Prediction: FAKE IMAGE")
