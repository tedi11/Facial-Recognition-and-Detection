from Parameters import *
import numpy as np
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import polynomial_kernel
import glob
import os
import cv2 as cv
import pdb
import pickle
import ntpath
from sklearn.svm import SVC
from copy import deepcopy
import timeit
from skimage.feature import hog
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset

class SimpleCNN_Recognition(nn.Module):
    def __init__(self, num_classes=5):  # Default to 2 classes
        super(SimpleCNN_Recognition, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(2, 2)


        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Activation function
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))

        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def eval_train(self, device, training_examples, train_labels):
        """
        Evaluate the classifier's performance on the training set.
        """
        self.eval()  # Switch to evaluation mode
        correct = 0  # Counter for correctly classified samples

        with torch.no_grad():
            # Convert examples and labels to tensors
            inputs = torch.tensor(training_examples, dtype=torch.float32).to(device)
            labels = torch.tensor(train_labels, dtype=torch.long).to(device)

            # Forward pass through the model
            outputs = self(inputs)  # Raw outputs (logits)

            # Compute probabilities using softmax
            probabilities = torch.softmax(outputs, dim=1)

            # Predicted class is the one with the highest probability
            predictions = torch.argmax(probabilities, dim=1)

            # Count correctly classified samples
            correct = (predictions == labels).sum().item()

            # Calculate accuracy
            accuracy = correct / len(labels)

            print(f"Accuracy on training data: {accuracy * 100:.2f}%")

            # Optionally, you can plot the class distribution of predictions
            predicted_classes = predictions.cpu().numpy()
            true_classes = labels.cpu().numpy()

            # Plotting the confusion matrix
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

            cm = confusion_matrix(true_classes, predicted_classes, labels=range(5))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                          display_labels=['dad', 'deedee', 'dexter', 'mom', 'nothing'])
            disp.plot(cmap=plt.cm.Blues)
            plt.title("Confusion Matrix on Training Data")
            plt.show()



def show_image(title, image):
    image = cv.resize(image, (0, 0), fx=5, fy=5)
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def intersection_over_union( bbox_a, bbox_b):
    x_a = max(bbox_a[0], bbox_b[0])
    y_a = max(bbox_a[1], bbox_b[1])
    x_b = min(bbox_a[2], bbox_b[2])
    y_b = min(bbox_a[3], bbox_b[3])

    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
    box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    return iou

transform = transforms.Compose([
    transforms.RandomRotation(10),  # Randomly rotate images by ±10 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color adjustments
    transforms.ToTensor()  # Convert PIL image to tensor
])

class CustomImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Apply the transform if specified
        if self.transform:
            image = self.transform(image)

        return image, label


class FacialRecognition:
    def __init__(self, params:Parameters):
        self.params = params
        self.best_model = None

    def train_classifier(self, training_examples, train_labels):
        cnn_file_name = os.path.join(self.params.dir_save_files, 'best_cnn_model_characters.pth')

        # Check if a trained model exists
        if os.path.exists(cnn_file_name):
            self.best_model = torch.load(cnn_file_name, weights_only=True)
            return

        # Prepare data
        batch_size = 64
        num_classes = len(set(train_labels))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Convert data to PyTorch tensors and create DataLoader
        train_dataset = CustomImageDataset(torch.tensor(training_examples, dtype=torch.float32),
                                      torch.tensor(train_labels, dtype=torch.long))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Define model, loss function, and optimizer
        model = SimpleCNN_Recognition(num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        num_epochs = 30
        best_accuracy = 0.0
        accuracy_list = []  # List to store accuracy per epoch

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)  # Add channel dimension if needed
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # Calculate accuracy
            accuracy = correct / total
            accuracy_list.append(accuracy)  # Append accuracy for this epoch
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / total:.4f}, Accuracy: {accuracy:.4f}")

            # Save the best model

            torch.save(model.state_dict(), cnn_file_name)

            if (epoch+1) % 3 == 0:
                #print("aa")
                model.eval_train(device=device, training_examples=training_examples, train_labels=train_labels)

        self.best_model = model

        # Visualize training accuracy
        plt.plot(range(1, num_epochs + 1), accuracy_list, label='Training Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy vs Epochs')
        plt.legend()
        plt.show()

