import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights, vit_b_16
from torchvision import models
from model.SingleImageDataset import SingleImageDataset

class Model():
    def __init__(self):
        self.model = self.load_model()

    # Function to load the model
    def load_model(self):
        # Check if CUDA is available and set PyTorch to use GPU or CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Set the random seed ( dot the same for cuda as well)
        RANDOM_SEED=42
        # Set seed for PyTorch
        torch.manual_seed(seed=RANDOM_SEED)
        # Setting the seed for CUDA (if using GPUs)
        torch.cuda.manual_seed(RANDOM_SEED)

        # Create the model
        model_rsn = models.resnet18(weights='IMAGENET1K_V1')
        # get the number of connections from the fully-connected layer
        num_ftrs = model_rsn.fc.in_features
        # the dataset has 4 classes, this is the number of outputs
        model_rsn.fc = nn.Linear(num_ftrs, 4)

        # Load the pretrained model
        model_rsn.load_state_dict(torch.load('./model/prj_resnet_20e_opt3.pth', map_location=torch.device(device), weights_only=True))
        
        # Move the model to the appropriate device
        model_rsn.to(device)

        return model_rsn

    # Function to return the prediction of a single image
    def single_prediction(self, dataloader):
        # Create an iterator from the testloader
        dataiter = iter(dataloader)

        # Retrieve the first batch of images
        images = next(dataiter)

        # Extract a single image 
        single_image = images[0]

        # Move the image to the GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        single_image = single_image.to(device)

        # Perform the prediction
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            single_image = single_image.unsqueeze(0)  # Add batch dimension
            output = self.model(single_image)
            _, predicted = torch.max(output, 1)

        return predicted.item()

    # Function to predict the output of a single image
    def predict_single_image(self, image_path):
        # Class names
        class_names = ['Class D', 'Class C', 'Class B', 'Class A']
        
        # Define transformations for the data
        transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

        # Create the dataset and dataloader
        dataset = SingleImageDataset(image_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        # Get the prediction
        result = self.single_prediction(dataloader)

        return class_names[result]