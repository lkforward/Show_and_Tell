# Show_and_Tell
This project is an implementation of google's image captioning neural network.

## Data Source
This project utilizes the COCO dataset (Common Objects in Context) dataset, which is "commonly used to train 
and benchmark object detection, segmentation, and captioning algorithms."
The notebook, 0_Dataset.ipynb, describes how to uses the COCO API to access the dataset. 

## File Descriptions
- model.py: Define the CNN encoder class and the LSTM decoder class using pyTorch. 
- Training.ipynb: train the CNN-LSTM model.
- Inference.ipynb: Use the trained model to generate caption for images. 

## References
This project is inspired by course project "Image Captioning" in Udacity Computer Vision Nanodegree program. 
