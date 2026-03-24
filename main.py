import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from pathlib import Path

# =========================
# CONFIGURATION
# =========================
BASE_DIR = Path(__file__).resolve().parent

TRAIN_DIR = BASE_DIR / "Training"
TEST_DIR  = BASE_DIR / "Testing"


IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 5
LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================
# TRANSFORMS
# =========================
def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    test_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    return train_tf, test_tf


# =========================
# DATALOADERS
# =========================
def get_dataloaders(train_tf, test_tf):
    train_ds = datasets.ImageFolder(TRAIN_DIR, train_tf)
    test_ds  = datasets.ImageFolder(TEST_DIR, test_tf)

    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0   # Windows-safe
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return train_dl, test_dl


# =========================
# MODEL
# =========================
def get_model():
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Flatten(),
        nn.Linear(128 * 8 * 8, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES)
    )

    return model.to(device)


# =========================
# TRAINING
# =========================
def train_model(model, train_dl, test_dl, optimizer, loss_fn):
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_dl:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = 100 * correct / total

        # ---------- VALIDATION ----------
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in test_dl:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_acc = 100 * val_correct / val_total

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Loss: {running_loss/len(train_dl):.4f} "
            f"Train Acc: {train_acc:.2f}% "
            f"Val Acc: {val_acc:.2f}%"
        )


# =========================
# INTERACTIVE PREDICTION
# =========================
def predict_by_index(model, test_dl):
    model.eval()
    classes = test_dl.dataset.classes

    while True:
        idx = int(input(
            f"\nEnter image index (0 to {len(test_dl.dataset)-1}) "
            f"or -1 to exit: "
        ))

        if idx == -1:
            break

        if idx < 0 or idx >= len(test_dl.dataset):
            print("Invalid index.")
            continue

        img, label = test_dl.dataset[idx]

        with torch.no_grad():
            img_processed = img.unsqueeze(0).to(device)
            outputs = model(img_processed)
            pred = outputs.argmax(dim=1).item()

        # De-normalize for display
        unnorm = img * 0.5 + 0.5
        unnorm = unnorm.clamp(0, 1)

        pil_img = to_pil_image(unnorm)
        pil_img = pil_img.resize((256, 256))

        plt.imshow(pil_img, interpolation='nearest')
        plt.axis('off')
        plt.title(f"True: {classes[label]} | Predicted: {classes[pred]}")
        plt.show()


# =========================
# MAIN
# =========================
def main():
    train_tf, test_tf = get_transforms()
    train_dl, test_dl = get_dataloaders(train_tf, test_tf)

    model = get_model()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    loss_fn = nn.CrossEntropyLoss()

    train_model(model, train_dl, test_dl, optimizer, loss_fn)

    # SAVE MODEL
    torch.save(model.state_dict(), "tumor_model.pth")
    print("Model saved as tumor_model.pth")

    predict_by_index(model, test_dl)


if __name__ == '__main__':
    main()
