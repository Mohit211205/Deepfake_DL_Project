import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import numpy as np


def evaluate_model(model, loader):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.cuda()
            y = y.cuda()

            outputs = model(x)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    cm = confusion_matrix(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)

    print("Pred counts:", np.bincount(all_preds))
    print("Label counts:", np.bincount(all_labels))

    print("\nConfusion Matrix:")
    print(cm)

    print("\nF1 Score:")
    print(f1)

    print("\nClassification Report:")
    print(report)
