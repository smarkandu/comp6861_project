import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import SpeechAIDataset
from model import AudioMLP


HUMAN_DIR = "./data/human"
SYNTHETIC_DIR = "./data/synthetic"

BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-3
SEED = 42


def evaluate(model, loader, device):
    model.eval()

    total = 0
    correct = 0
    total_loss = 0.0

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            preds = (torch.sigmoid(logits) >= 0.5).float()

            total_loss += loss.item() * x.size(0)
            correct += (preds == y).sum().item()
            total += x.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def main():
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = SpeechAIDataset(HUMAN_DIR, SYNTHETIC_DIR)

    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    sample_x, _ = dataset[0]
    input_dim = sample_x.shape[0]

    model = AudioMLP(input_dim=input_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0
        total = 0
        correct = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(logits) >= 0.5).float()

            running_loss += loss.item() * x.size(0)
            correct += (preds == y).sum().item()
            total += x.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )


if __name__ == "__main__":
    main()