import os
import torch
from PIL import Image
from torchvision import transforms
from models.deepfake_classifier import DeepfakeClassifier


def predict_folder():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # fixed folder path
    folder_path = "my_test_images"

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

    valid_ext = (".jpg", ".jpeg", ".png")

    real_count = 0
    fake_count = 0

    print("\n===== IMAGE PREDICTIONS =====\n")

    for file_name in os.listdir(folder_path):
        if not file_name.lower().endswith(valid_ext):
            continue

        image_path = os.path.join(folder_path, file_name)

        img = Image.open(image_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img)
            pred = torch.argmax(output, dim=1).item()

        if pred == 0:
            label = "REAL"
            real_count += 1
        else:
            label = "FAKE"
            fake_count += 1

        print(f"{file_name} --> {label}")

    print("\n===== FINAL SUMMARY =====")
    print(f"Total REAL images : {real_count}")
    print(f"Total FAKE images : {fake_count}")
    print(f"Total images      : {real_count + fake_count}")


if __name__ == "__main__":
    predict_folder()
