import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2


# Define the ConvNet class (same as the original code)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Function to process the image from the canvas
def process_image(image_data):
    # Convert from RGBA to grayscale
    img = cv2.cvtColor(image_data, cv2.COLOR_RGBA2GRAY)
    # Resize to 28x28 (as in MNIST)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img = Image.fromarray(img)
    img = transform(img).unsqueeze(0)  # Add batch dimension
    return img


# Create the Streamlit interface
st.title("Handwriting Recognition with Pytorch")
st.markdown("Draw a digit (0-9) on the canvas and press 'Predict' to make a prediction!")

# Create the canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvNet().to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# Predict button
if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Process the image
        img = process_image(canvas_result.image_data)
        img = img.to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(img)
            probabilities = torch.softmax(outputs, dim=1)  # Convert logits to probabilities
            confidence, predicted = torch.max(probabilities, 1)
            predicted_digit = predicted.item()
            confidence_score = confidence.item() * 100  # Convert to percentage

        # Display the results
        st.write(f"**Predicted Digit**: {predicted_digit}")
        st.write(f"**Confidence Score**: {confidence_score:.2f}%")
    else:
        st.write("Please draw a digit before predicting!")