import torch
from torch.utils.data import DataLoader
from dataset import UltrasoundDataset
from unet import UNet

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = UltrasoundDataset("clean_dataset/images", "clean_dataset/masks")

loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = UNet().to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 1010

for epoch in range(EPOCHS):
    total_loss = 0

    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)

        pred = model(img)
        loss = criterion(pred, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "unet_model.pth")
print("✅ MODEL SAVED")