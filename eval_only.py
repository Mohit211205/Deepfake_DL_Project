import torch
from torch.utils.data import DataLoader
from datasets_loader.deepfake_dataset import DeepfakeDataset
from models.deepfake_classifier import DeepfakeClassifier
from trainers.evaluate import evaluate_model

dataset = DeepfakeDataset(
    "datasets/ffpp/real_frames",
    "datasets/ffpp/fake_frames"
)

loader = DataLoader(dataset, batch_size=16, shuffle=False)

model = DeepfakeClassifier().cuda()
model.load_state_dict(torch.load("outputs/checkpoints/classifier.pth"))

evaluate_model(model, loader)
