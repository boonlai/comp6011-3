import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm


def load_backbone(device):
    """
    Load the DINOv2 backbone.
    """
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone.to(device)

    return backbone


class ECGClassifier(nn.Module):
    """
    A simple MLP classifier that uses the DINOv2 backbone.
    """

    # DINOv2 has an embedding dimension of 384 for ViT-S according to the model card:
    # https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md
    def __init__(self, backbone, n_classes, embed_dim=384, hidden_dim=256):
        super().__init__()

        self.backbone = backbone
        self.fc1 = nn.Linear(embed_dim, hidden_dim)  # Single hidden layer
        self.relu = nn.ReLU(True)  # ReLU activation
        self.fc2 = nn.Linear(hidden_dim, n_classes)  # Output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the classifier.
        """
        emb = self.backbone(x)

        # Weird Tuple return, just cast it
        if isinstance(emb, (list, tuple)):
            emb = emb[0]

        h = self.relu(self.fc1(emb))  # 384 -> 256
        out = self.fc2(h)  # 256 -> n_classes (7)

        return out


def get_dataloaders(train_dir, val_dir, batch=64, workers=2):
    """
    Get the dataloaders for the training and validation sets.
    """
    # Check if running on Windows and set workers to 0 to avoid multiprocessing issues
    if os.name == "nt":
        workers = 0

    tfm = transforms.Compose(
        [
            transforms.Resize(518),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    # Format: data/[train|valid]/[class]/[image]
    train_ds = datasets.ImageFolder(train_dir, tfm)
    val_ds = datasets.ImageFolder(val_dir, tfm)

    t_loader = DataLoader(
        train_ds, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=True
    )
    v_loader = DataLoader(
        val_ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True
    )

    return t_loader, v_loader


def train_model(model, train_loader, val_loader, device, epochs=14, lr=1e-6, ckpt=None):
    """
    Train the head of the model.
    """
    model_params = model.parameters()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optim = Adam(model_params, lr=lr)
    scaler = GradScaler(device)
    best = 0.0

    # Loop over epochs
    for ep in range(epochs):
        model.train()
        model.backbone.train()

        # Show training progress
        bar = tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs}", leave=False)
        running = 0.0

        for x, y in bar:
            x, y = x.to(device), y.to(device)
            # Use mixed precision training
            optim.zero_grad()
            with autocast(device.type):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            # loss.backward()
            # optim.step()
            running += loss.item() * x.size(0)

            # Include loss for visualization
            bar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation step
        model.eval()
        model.backbone.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        val_acc = correct / total

        print(
            f"Epoch {ep+1:2d}  "
            f"train loss {running/len(train_loader.dataset):.4f}  "
            f"val acc {val_acc:.3%}"
        )

        # Given the checkpoint, save the model if the validation accuracy is better
        if ckpt and val_acc > best:
            torch.save(model.state_dict(), ckpt)
            best = val_acc

    print(f"Best val acc: {best:.3%}")


def get_finetuned_model(
    device, n_classes, *, checkpoint=None, train=False
) -> nn.Module:
    """
    Get the finetuned model, or train one if specified or the checkpoint doesn't exist.
    """
    ckpt = checkpoint or f"finetuned_model_{n_classes}.pt"
    model = ECGClassifier(load_backbone(device), n_classes).to(device)

    if not train or os.path.exists(ckpt):
        print(f"Loading fine-tuned weights from {ckpt}")
        model.load_state_dict(torch.load(ckpt, map_location=device))
    elif train:
        print("Fine-tuning DINOv2 …")
        t_loader, v_loader = get_dataloaders("data/train", "data/valid")
        train_model(model, t_loader, v_loader, device, epochs=14, lr=1e-6, ckpt=ckpt)
    else:
        raise ValueError("The fine-tune model has not been trained yet.")

    model.eval()
    model.backbone.eval()
    return model
