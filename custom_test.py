import torch
from PIL import Image
from torchvision import transforms
from models.deepfake_classifier import DeepfakeClassifier


def predict_image(image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    model = DeepfakeClassifier().to(device)
    model.load_state_dict(
        torch.load(
            "outputs/checkpoints/classifier.pth",
            map_location=device
        )
    )
    model.eval()

    # transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # load image
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    # predict
    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    confidence = probs[0][pred].item() * 100

    if pred == 0:
        print(f"Prediction: REAL IMAGE ({confidence:.2f}%)")
    else:
        print(f"Prediction: FAKE / AI GENERATED IMAGE ({confidence:.2f}%)")


if __name__ == "__main__":
    path = input("Enter image path: ")
    predict_image(path)
