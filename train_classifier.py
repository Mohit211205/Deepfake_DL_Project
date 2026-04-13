import torch
from torch.optim import Adam
from tqdm import tqdm
from models.deepfake_classifier import DeepfakeClassifier
import os


def train_classifier(loader, epochs=5):
    model = DeepfakeClassifier().cuda()

    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()

        total_loss = 0
        correct = 0
        total = 0

        for x, y in tqdm(loader):
            x = x.cuda()
            y = y.cuda()

            outputs = model(x)

            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        acc = (correct / total) * 100
        avg_loss = total_loss / len(loader)

        print(f"Epoch {epoch+1}")
        print(f"Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")

    os.makedirs("outputs/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "outputs/checkpoints/classifier.pth")

    print("✅ Classifier saved successfully")

    return model
