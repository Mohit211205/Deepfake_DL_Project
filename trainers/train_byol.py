import torch
from torch.optim import Adam
from tqdm import tqdm

from models.byol_pretrain import BYOL
from models.losses import byol_loss


def train_byol(loader, epochs=5):
    model = BYOL().cuda()
    optimizer = Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x1, x2 in tqdm(loader):
            x1 = x1.cuda()
            x2 = x2.cuda()

            pred, target = model(x1, x2)

            loss = byol_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()

            # ✅ gradient clipping (stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            # ✅ EMA update for target encoder (VERY IMPORTANT)
            for param_q, param_k in zip(
                model.online_encoder.parameters(),
                model.target_encoder.parameters()
            ):
                param_k.data = 0.996 * param_k.data + 0.004 * param_q.data

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # ✅ save model
    torch.save(model.state_dict(), "outputs/checkpoints/byol_pretrained.pth")
    print("Model saved successfully!")
