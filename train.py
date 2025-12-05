import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

"""
git clone https://github.com/Emilmrk/preprocessed-facial-emotions-224
"""

data_dir = "preprocessed-facial-emotions-224/datasetpreprocesado"

# transforms
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
full_ds = torchvision.datasets.ImageFolder(root=data_dir, transform=train_transform)

# train/test split:
train_size = int(0.8 * len(full_ds))
val_size = len(full_ds) - train_size
train_ds, val_ds = random_split(full_ds, [train_size, val_size])

# Use validation transform for val split
val_ds.dataset.transform = test_transform

train_loader = DataLoader(train_ds, shuffle=True, batch_size=32)
val_loader = DataLoader(val_ds, shuffle=True, batch_size=32)

# Model 
class CNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
        self.lrn1 = nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.lrn2 = nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool1(self.lrn1(nn.functional.relu(self.conv1(x))))
        x = self.pool2(self.lrn2(nn.functional.relu(self.conv2(x))))

        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = self.pool3(nn.functional.relu(self.conv5(x)))

        x = torch.flatten(x, 1)
        x = self.drop1(nn.functional.relu(self.fc1(x)))
        x = self.drop2(nn.functional.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    model = CNN().to(device)

    LEARNING_RATE = 0.01
    EPOCHS = 30

    # To account for the dataset being uneven and skewed towards certain classes, we'll use weighted classes
    # https://towardsdatascience.com/how-to-handle-imbalance-data-and-small-training-sets-in-ml-989f8053531d/
    counts = torch.tensor([2174, 624, 1587, 5200, 6275, 1696, 1549], dtype=torch.float)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)

    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = optim.SGD(lr=LEARNING_RATE, params=model.parameters(), momentum=0.9)

    # TODO - calculate the final test accuracy
    train_acc_history = []
    val_acc_history = []

    train_loss_history = []
    val_loss_history = []

    print("Training model...")
    model = CNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9) # best paramters from hyperparameter grid

    for epoch in range(EPOCHS): # full train 30 epochs
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = preds.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc  = correct / total

        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                preds = model(images)
                loss = criterion(preds, labels)
                val_loss += loss.item()

                _, predicted = preds.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total

        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        print(f"Epoch {epoch+1}/{EPOCHS} \n"
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}\n"
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save model
    torch.save(model.state_dict(), "model.pth")

    plt.figure(figsize=(10,5))
    plt.plot(train_acc_history, label="Train Accuracy")
    plt.plot(val_acc_history, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

