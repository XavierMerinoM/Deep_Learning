# Banana Ripeness Classifier

Banana ripeness classification is a common problem in agriculture and food supply chains. This project leverages deep learning algorithms to classify bananas into different stages of ripeness based on their physical characteristics (e.g., color and texture).
This project takes the paper [Banana Ripeness Level Classification using a Simple CNN Model Trained with Real and Synthetic Datasets](https://arxiv.org/pdf/2504.08568) as a basis. However, the main objective of the present project is not to replicate the pipeline of the paper but to implement a different approach.

## Dataset
The dataset used for this project includes images of bananas at various stages of ripeness. Key details:
- **Classes**: A, B, C, D; corresponding to A to the lower level and D to the highest level of ripeness
- **Source**: The full dataset for the project can be downloaded from [BananaRipeness](https://github.com/luischuquim/BananaRipeness) repository. The present model was trained with a subset of 7,000 images.
- **Preprocessing**: Images were resized to 16x16 and normalized for training.

## Model Architecture
The model is based on Deep Residual Learning for Image Recognition. This is implemented via pretrained library and weights, [resnet18](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html#torchvision.models.resnet18) and IMAGENET1K_V1, respectively.

## Training Process
- **Framework**: PyTorch.
- **Loss Function**: Crossentropy.
- **Optimizer**: Adam.
- **Metrics**: Accuracy, Precision, Recall, and F1-Score.
- **Epochs**: 5, 10, 20.

The training process includes weight decay to prevent overfitting and auto-cast to speed up the training process.

## Results
The model achieved the following performance metrics:
- **Training Accuracy**: 94.90%
- **Validation Accuracy**: 94.30%
- **Test Accuracy**: 95.40%

The full metrics for the test are:
- **Precision**: 96.40%
- **Recall**: 95.40%
- **F1-score**: 95.86%

Visualization of the results, such as sample predictions, can be found in the Jupyter Notebook from the notebook folder.

## Extra

A testing interface was implemented using streamlit library to verify the classification with new banana images. The result is the class of the image with the attention region of the algorithm. In addition, the user can send a prompt to ChatGPT to get more information about the result. It is necessary to add an api-key.txt file to tools folder with the API key to get access to ChatGPT. 
The Application folder contains all the Python code, including the pretrained model.

## Versioning

- **Version 1**: Initial model, including Jupyter Notebook and Python implementation.
