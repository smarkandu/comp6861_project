import os
import torch

from models.eend import TinyEEND
from models.eend_loss import pit_bce_loss
from debug import vprint

def train_eend(
    train_loader,
    val_loader,
    input_dim=80,
    num_speakers=4,
    hidden_dim=128,
    num_layers=2,
    num_heads=4,
    dropout=0.1,
    epochs=10,
    lr=1e-4,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    save_path="outputs/models/eend_best.pt",
    patience=5,
    min_delta=1e-4,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    vprint("=== Initializing TinyEEND ===")

    model = TinyEEND(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_speakers=num_speakers,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    vprint(f"[INFO] Device: {device}")
    vprint(f"[INFO] Save path: {save_path}")
    vprint(f"[INFO] Early stopping patience: {patience}")
    vprint(f"[INFO] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        vprint(f"\n=== Epoch {epoch}/{epochs} ===")

        for i, batch in enumerate(train_loader):
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            padding_mask = batch.get("padding_mask", None)

            if padding_mask is not None:
                padding_mask = padding_mask.to(device)

            if epoch == 1 and i == 0:
                vprint(f"[DEBUG] features shape: {features.shape}")
                vprint(f"[DEBUG] targets shape: {targets.shape}")
                if padding_mask is not None:
                    vprint(f"[DEBUG] padding_mask shape: {padding_mask.shape}")

            optimizer.zero_grad()

            logits = model(features, padding_mask=padding_mask)

            if epoch == 1 and i == 0:
                vprint(f"[DEBUG] logits shape: {logits.shape}")

            loss = pit_bce_loss(logits, targets, padding_mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()

            if i % max(1, len(train_loader) // 5) == 0:
                vprint(f"[Train] Batch {i + 1}/{len(train_loader)} | loss={loss.item():.4f}")

        avg_train_loss = total_loss / max(1, len(train_loader))

        vprint("[INFO] Running validation...")
        val_loss = evaluate_eend_loss(model, val_loader, device)

        vprint(f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_loss:.4f}"
        )

        improved = val_loss < (best_val_loss - min_delta)

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "input_dim": input_dim,
                    "num_speakers": num_speakers,
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "dropout": dropout,
                    "lr": lr,
                },
                save_path,
            )

            vprint(f"[INFO] Saved new best model → {save_path}")
        else:
            epochs_without_improvement += 1
            vprint(f"[INFO] No improvement for "
                f"{epochs_without_improvement}/{patience} epoch(s)"
            )

        if epochs_without_improvement >= patience:
            vprint(f"[INFO] Early stopping triggered at epoch {epoch}. "
                f"Best epoch was {best_epoch} with val_loss={best_val_loss:.4f}"
            )
            break

    vprint("=== Training complete ===")
    vprint(f"[INFO] Best epoch: {best_epoch}")
    vprint(f"[INFO] Best val_loss: {best_val_loss:.4f}")
    vprint(f"[INFO] Best model saved at: {save_path}")
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model, best_val_loss

@torch.no_grad()
def evaluate_eend_loss(model, val_loader, device):
    model.eval()
    total_loss = 0.0

    for i, batch in enumerate(val_loader):
        features = batch["features"].to(device)
        targets = batch["targets"].to(device)
        padding_mask = batch.get("padding_mask", None)

        if padding_mask is not None:
            padding_mask = padding_mask.to(device)

        logits = model(features, padding_mask=padding_mask)
        loss = pit_bce_loss(logits, targets, padding_mask)

        total_loss += loss.item()

        vprint(f"[Val] Batch {i + 1}/{len(val_loader)} | loss={loss.item():.4f}")

    return total_loss / max(1, len(val_loader))


from datasets.ami import AMIDataset
from datasets.eend_dataset import create_eend_dataloaders

def main():
    recording_audio_dir = "./data/amicorpus"
    ami = AMIDataset(
        audio_dir=recording_audio_dir,
        annotation_dir="./data/ami_public_manual_1.6.2",
        target_sr=16000,
    )

    train_loader, val_loader, test_loader = create_eend_dataloaders(
        ami_dataset=ami,
        train_recordings=["ES2002a", "ES2002b"],
        val_recordings=["ES2003a"],
        test_recordings=["ES2004a"],
        sample_rate=16000,
        n_mels=80,
        hop_length=160,
        num_speakers=4,
        batch_size=1,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, best_val_loss = train_eend(
        train_loader,
        val_loader,
        input_dim=80,
        num_speakers=4,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        epochs=30,
        lr=3e-4,
        device=device,
        save_path="./models/eend_best.pt",
        patience=5,
    )

if __name__ == "__main__":
    main()