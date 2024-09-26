from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from rembg import remove
from torchvision import datasets
from torch.utils.data import DataLoader
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define hyperparameters
input_size = 200
num_classes = 3
num_epochs = 20
batch_size = 32
learning_rate = 0.01


# Define transforms
transform = transforms.Compose([
    transforms.Resize(250),
    transforms.CenterCrop((input_size, input_size)),
    transforms.RandomHorizontalFlip(p=0.5), # 50% horizontal flip
    transforms.RandomVerticalFlip(p=0.5), # 50% vertical flip
    transforms.RandomGrayscale(p=1), # 100% convert to gray
    transforms.RandomRotation(degrees=(-10, 10)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride

        # Adjust the dimensions if the input/output channels or stride changes
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out
    
# Define the ResNet model
class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self.make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self.make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self.make_layer(256, 512, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x



def main():
    import streamlit as st
    
    st.title("Lonely PRS Demo")
    model = torch.load('remBG.mdl', map_location=torch.device('cpu'))
    model.eval()
    labels_dict = {0: 'paper', 1: 'rock', 2: 'scissors'}    
    imgRock = Image.open("robotTurn/rock.png")
    imgScissors = Image.open("robotTurn/scissors.png")
    imgPaper = Image.open("robotTurn/paper.png")


    st.subheader("It's your turn. Take a photo!")
    img_file_buffer = st.camera_input("")

    transform = transforms.Compose([
        transforms.Resize(250),
        transforms.CenterCrop((input_size, input_size)),
        transforms.RandomGrayscale(p=1), # 100% convert to gray
        transforms.ToTensor(),  # Convert image to tensor
    ])
    if img_file_buffer:
        # Read image and display
        image = Image.open(img_file_buffer)

        # Predict the gesture
        rem_image = remove(image)
        image = rem_image.convert('RGB')

        # Apply transformations to the image
        input_array = transform(image)
        input_tensor = input_array.unsqueeze(0)
        input_array = input_array.permute(1, 2, 0).numpy() 
    
        # Make predictions using the model
        with torch.no_grad():
            output = model(input_tensor)

        # Display the prediction
        predicted_class = torch.argmax(output).item()
        st.write(f"Predicted Gesture: {labels_dict[predicted_class]}")

        import streamlit as st
        import numpy as np

        st.subheader("My show time😁")
        # Generate a random number from 0, 1, or 2
        random_number = np.random.randint(0, 3)
        rd_gesture = labels_dict[random_number]

        # Display the result
        st.write(f"My Turn: {rd_gesture} !")

        if (random_number == 0):
            st.image(imgPaper, use_column_width=True)
        elif (random_number == 1):
            st.image(imgRock, use_column_width=True)
        else:
            st.image(imgScissors, use_column_width=True)

        def rps(user, robot):
          if (user==robot):
            st.warning("It's a draw!")
          elif (user == 0 and robot == 2):
            st.warning("You Lose.")
          elif (robot == 0 and user == 2):
            st.success("You Win.")
          elif (user < robot):
              st.success("You Win.")
          elif (user > robot):
              st.warning("You Lose.")
        
        rps(predicted_class, random_number)

          
        st.subheader("More Details on Image Preprocessing")

        st.image(rem_image, caption='BG-Removed Image', use_column_width=True)
        st.image(input_array, caption='Processed Image', use_column_width=True)


# Run the app
if __name__ == "__main__":
    main()
