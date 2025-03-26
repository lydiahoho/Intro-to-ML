import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
import os
import pandas as pd
from PIL import Image

# Set GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pretrained model
model_path = "./records/FGVC-HERBS/basline/backup/best.pt"
checkpoint = torch.load(model_path)
model = checkpoint["model"]  # Extract the model from the dictionary
model = model.to(device)
model.eval()

# Set the test data folder path
test_folder = "./dataset/test"

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create the test dataset
test_dataset = ImageFolder(test_folder, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Initialize the results list
results = []

# Make predictions
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = Variable(inputs)
        inputs = inputs.to(device)

        # Model prediction
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        # Append the result to the results list
        results.append(predicted.item())

# Create a DataFrame
df = pd.DataFrame({"id": [os.path.basename(path) for path, _ in test_loader.dataset.imgs],
                   "label": results})

# Save the results to a CSV file
df.to_csv("./predict_result/predictions_model1.csv", index=False)

