import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class_names = [
    "angry", "disgust", "fear", "happy", "neutral",
    "sad", "surprise"
]

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

model = CNN().to(device)

state_dict = torch.load("model.pth", map_location=device)
model.load_state_dict(state_dict)

model.eval()

def infer(PIL_image):
    transform = transforms.Compose([
        transforms.Resize((228, 228)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    t_image = transform(PIL_image)
    t_image = t_image.unsqueeze(0)
    t_image = t_image.to(device)

    model.eval()
    with torch.no_grad():
        pred = model(t_image).argmax(dim=1).item()
        return class_names[pred]

