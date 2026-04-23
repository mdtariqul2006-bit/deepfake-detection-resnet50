from model import imageModel
import copy
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm


class DeepfakeDataset(Dataset):
    def __init__(self, fake_dir, real_dir, transform=None):
        self.transform = transform
        self.samples   = []

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for fname in os.listdir(fake_dir):
            if os.path.splitext(fname)[1].lower() in exts:
                self.samples.append((os.path.join(fake_dir, fname), 0))  # 0 = fake

        for fname in os.listdir(real_dir):
            if os.path.splitext(fname)[1].lower() in exts:
                self.samples.append((os.path.join(real_dir, fname), 1))  # 1 = real

        print(f"Fake: {sum(1 for _, l in self.samples if l == 0):,} | "
              f"Real: {sum(1 for _, l in self.samples if l == 1):,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, label


if __name__ == "__main__":

    NUM_CLASSES   = 2
    NUM_EPOCHS    = 16
    BATCH_SIZE    = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY  = 1e-4
    MODEL_SAVE    = "best_model.pth"

    FAKE_DIR     = "processedImages/fakeProcessedImages"
    REAL_DIR     = "processedImages/realProcessedImages"


    VAL_FAKE_DIR = "processedImages/processedImagesValidation/validationFake"
    VAL_REAL_DIR = "processedImages/processedImagesValidation/validationReal"


    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(" Using Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")


    model, transforms = imageModel(num_classes=NUM_CLASSES)
    model = model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ResNet50 — trainable params: {trainable:,}")

  
    print("\nLoading training data:")
    train_dataset = DeepfakeDataset(FAKE_DIR, REAL_DIR, transform=transforms)

    print("Loading validation data:")
    val_dataset   = DeepfakeDataset(VAL_FAKE_DIR, VAL_REAL_DIR, transform=transforms)

    print(f"\n Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)



    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)



    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(NUM_EPOCHS):

       
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            t_loss    += loss.item() * images.size(0)
            t_correct += outputs.argmax(1).eq(labels).sum().item()
            t_total   += labels.size(0)

        scheduler.step()

    
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]  "):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss    = criterion(outputs, labels)

                v_loss    += loss.item() * images.size(0)
                v_correct += outputs.argmax(1).eq(labels).sum().item()
                v_total   += labels.size(0)

        print(f"Epoch {epoch+1:>2}/{NUM_EPOCHS}  "
              f"train loss: {t_loss/t_total:.4f}  acc: {t_correct/t_total:.4f}  │  "
              f"val loss: {v_loss/v_total:.4f}  acc: {v_correct/v_total:.4f}")

        if v_correct / v_total > best_val_acc:
            best_val_acc = v_correct / v_total
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, MODEL_SAVE)
            print(f" Best model saved → {MODEL_SAVE}  (val acc: {best_val_acc:.4f})")

    print(f"\n Training complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Model accuracy in percentage: {best_val_acc * 100:.2f}%")