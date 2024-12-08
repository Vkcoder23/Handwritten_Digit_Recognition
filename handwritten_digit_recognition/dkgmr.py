import torch
from app import DigitRecognitionModel  # Adjust the import based on your file structure

# Create a dummy input tensor with the shape [1, 1, 28, 28]
dummy_input = torch.randn(1, 1, 28, 28)

# Initialize the model
model = DigitRecognitionModel()

# Pass the dummy input through the model
with torch.no_grad():
    dummy_output = model(dummy_input)

# Print the output shape
print(f"Shape of dummy output: {dummy_output.shape}")
