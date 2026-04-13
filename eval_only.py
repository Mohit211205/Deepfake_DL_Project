import torch
from torch.utils.data import DataLoader, Subset
from datasets_loader.deepfake_dataset import DeepfakeDataset
from models.deepfake_classifier import DeepfakeClassifier
from trainers.evaluate import evaluate_model

# Load full dataset
dataset = DeepfakeDataset(
    "datasets/ffpp/real_frames",
    "datasets/ffpp/fake_frames"
)

# Total real images count
half = len(dataset.paths) // 2

# Balanced subset: 250 real + 250 fake
real_indices = list(range(250))
fake_indices = list(range(half, half + 250))

subset_indices = real_indices + fake_indices

dataset = Subset(dataset, subset_indices)

# Loader
loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False
)

# Load trained classifier
model = DeepfakeClassifier().cuda()
model.load_state_dict(
    torch.load("outputs/checkpoints/classifier.pth")
)

# Run evaluation
evaluate_model(model, loader)
