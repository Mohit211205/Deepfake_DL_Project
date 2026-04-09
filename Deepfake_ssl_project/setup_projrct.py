import os

project_structure = {
    "configs": {
        "config.py": ""
    },
    "datasets": {
        "faceforensics": {
            "train": {
                "real": {},
                "fake": {}
            },
            "test": {
                "real": {},
                "fake": {}
            }
        },
        "celebdf": {
            "real": {},
            "fake": {}
        }
    },
    "datasets_loader": {
        "deepfake_dataset.py": ""
    },
    "models": {
        "byol_encoder.py": "",
        "frequency_branch.py": "",
        "fusion_detector.py": "",
        "losses.py": ""
    },
    "trainers": {
        "train_byol.py": "",
        "train_detector.py": "",
        "evaluate_cross_dataset.py": ""
    },
    "utils": {
        "augmentations.py": "",
        "fft_utils.py": "",
        "metrics.py": ""
    },
    "outputs": {
        "checkpoints": {},
        "logs": {},
        "heatmaps": {}
    },
    "main.py": ""
}


def create_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)

        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            with open(path, "w") as f:
                f.write(content)


if __name__ == "__main__":
    create_structure(os.getcwd(), project_structure)
    print("Clean project structure created successfully!")