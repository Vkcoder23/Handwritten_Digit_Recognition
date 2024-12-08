from flask import Flask, render_template, request, jsonify
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import base64
import io


app = Flask(__name__)

class DigitRecognitionModel(nn.Module):
    def __init__(self):
        super(DigitRecognitionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self._calculate_conv_output_size()
        self.fc1 = nn.Linear(self.flattened_size, 128)
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

# Initialize the model
model = DigitRecognitionModel()

# Load the saved state_dict
model.load_state_dict(torch.load('C:\handwritten_vk\handwritten_digit_recognition\model\digit_recognition_model.pth', map_location=torch.device('cpu')))
model.eval()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Step 1: Decode the image data
        data = request.json['image']
        print(f"Raw data length: {len(data)}")

        image_data = base64.b64decode(data.split(',')[1])
        image = Image.open(io.BytesIO(image_data))

        # Debug: Show or save the image to verify
        image.show()  # Will display the image if running locally with GUI support
        image.save("debug_image.png")  # Save the image for inspection

        # Step 2: Convert to grayscale and resize to 28x28
        image = image.convert('L')  # Convert to grayscale
        image = image.resize((28, 28))

        # Step 3: Convert to numpy array and check the pixel range
        image = np.array(image).astype(np.float32)
        print(f"Numpy array shape: {image.shape}, min/max: {np.min(image)}, {np.max(image)}")

        # Check if the image is non-empty before normalization
        if np.min(image) == np.max(image):
            print("Warning: Image appears to be uniform, likely empty.")

        # Normalize the image to the range [-1, 1]
        image = (image / 255.0 - 0.5) / 0.5
        print(f"Normalized image min/max: {np.min(image)}, {np.max(image)}")

        # Step 4: Convert to torch tensor and add batch and channel dimensions
        image = torch.tensor(image).unsqueeze(0).unsqueeze(0)

        # Debugging information
        print(f"Input image shape: {image.shape}")

        # Step 5: Ensure the model is in evaluation mode and make prediction
        model.eval()
        with torch.no_grad():
            output = model(image)
            prediction = torch.argmax(output).item()

        # Debugging information
        print(f"Model output: {output}")
        print(f"Predicted class: {prediction}")

        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction'}), 500


if __name__ == '__main__':
    app.run(debug=True)
