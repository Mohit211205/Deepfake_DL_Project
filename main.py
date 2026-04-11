from torch.utils.data import DataLoader
from datasets_loader.deepfake_dataset import DeepfakeDataset
from trainers.train_classifier import train_classifier
from trainers.evaluate import evaluate_model

# ✅ Dataset
dataset = DeepfakeDataset(
    "datasets/ffpp/real_frames",
    "datasets/ffpp/fake_frames"
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4
)

# ✅ Train classifier
model = train_classifier(loader, epochs=5)

# ✅ Evaluate model
evaluate_model(model, loader)
