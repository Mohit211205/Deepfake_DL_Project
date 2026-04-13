from torch.utils.data import DataLoader, random_split
from datasets_loader.deepfake_dataset import DeepfakeDataset
from trainers.train_classifier import train_classifier
from trainers.evaluate import evaluate_model

# Load full dataset
dataset = DeepfakeDataset(
    "datasets/ffpp/real_frames",
    "datasets/ffpp/fake_frames"
)

# Split dataset: 80% train, 20% test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(
    dataset,
    [train_size, test_size]
)
print("Train size:", len(train_dataset))
print("Test size:", len(test_dataset))

# Train loader
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

# Test loader
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)
sample_x, sample_y = next(iter(train_loader))
print("Batch shape:", sample_x.shape)
print("Batch labels:", sample_y[:20])
# Train model
model = train_classifier(train_loader, epochs=5)

# Evaluate on unseen test data
evaluate_model(model, test_loader)
