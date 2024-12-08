import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class DigitRecognitionModel(nn.Module):
    def __init__(self):
        super(DigitRecognitionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        # Calculate size after conv and pooling layers
        self._calculate_conv_output_size()
        self.fc1 = nn.Linear(self.flattened_size, 128)  # Updated to match flattened size
        self.fc2 = nn.Linear(128, 10)

    def _calculate_conv_output_size(self):
        # Create a dummy tensor with the input size
        dummy_input = torch.zeros(1, 1, 28, 28)  # Batch size, channels, height, width
        output = self._forward_conv(dummy_input)
        self.flattened_size = output.numel()

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  # Flatten dynamically based on the batch size
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10

    # Data transformation and loading
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model, loss function, and optimizer
    model = DigitRecognitionModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in train_loader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), 'C:\handwritten_vk\handwritten_digit_recognition\model\digit_recognition_model.pth')
    print("Model saved to 'model/digit_recognition_model.pth'")

if __name__ == '__main__':
    main()
