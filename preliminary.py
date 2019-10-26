# This is adopted from "Notebook 1. Preliminaries"
#
# Objectives:
# 	Generate a dataloader with training data. 
#	Design a model architecture. 
#	Support an easy forward run (for salinity checking). 

import numpy as np
import torch
from torchvision import transforms
import torch.utils.data as data

import sys
sys.path.append('/opt/cocoapi/PythonAPI')
from pycocotools.coco import COCO

#!pip install nltk
import nltk
nltk.download('punkt')

from data_loader import get_loader
from model import EncoderCNN, DecoderRNN


# Define a transform to pre-process the training images.
transform_train = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

# Set the minimum word count threshold.
vocab_threshold = 5
# Specify the batch size.
batch_size = 10
# Specify the dimensionality of the image embedding.
embed_size = 256

data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_from_file=True)

# Store the size of the vocabulary.
vocab_size = len(data_loader.dataset.vocab)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def caption_length_summary(data_loader):
	# In the code cell below, we use this list to print the total number 
	# of captions in the training data with each length. As you will see 
	# below, the majority of captions have length 10. Likewise, very 
	# short and very long captions are quite rare.
	from collections import Counter

	# Tally the total number of training captions with each length.
	counter = Counter(data_loader.dataset.caption_lengths)
	lengths = sorted(counter.items(), key=lambda pair: pair[1], reverse=True)
	for value, count in lengths:
	    print('value: %2d --- count: %5d' % (value, count))

	return


def get_sample_batch(data_loader, do_print_data=False):
	# Randomly sample a caption length, and sample indices with that length.
	indices = data_loader.dataset.get_train_indices()
	print('sampled indices:', indices)

	# Create and assign a batch sampler to retrieve a batch with the sampled indices.
	new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
	data_loader.batch_sampler.sampler = new_sampler
	 
	images, captions = next(iter(data_loader))
	    
	print('images.shape:', images.shape)
	print('captions.shape:', captions.shape)

	if do_print_data:
		print('images:', images)
		print('captions:', captions)

	return images, captions


def embedding_sample_caption():
	pass


def check_encoder(images):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Initialize the encoder. (Optional: Add additional arguments if necessary.)
	encoder = EncoderCNN(embed_size)

	# Move the encoder to GPU if CUDA is available.
	encoder.to(device)
	# Move last batch of images (from Step 2) to GPU if CUDA is available.   
	images = images.to(device)

	# Pass the images through the encoder.
	features = encoder(images)

	print('type(features):', type(features))
	print('features.shape:', features.shape)

	# Check that your encoder satisfies some requirements of the project! :D
	assert type(features)==torch.Tensor, "Encoder output needs to be a PyTorch Tensor." 
	assert (features.shape[0]==batch_size) & (features.shape[1]==embed_size), "The shape of the encoder output is incorrect."

	return features


def check_decoder(features, captions):
	# Specify the number of features in the hidden state of the RNN decoder.
	hidden_size = 512

	# Initialize the decoder.
	decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

	# Move the decoder to GPU if CUDA is available.
	decoder.to(device)
	    
	# Move last batch of captions (from Step 1) to GPU if CUDA is available 
	captions = captions.to(device)

	# Pass the encoder output and captions through the decoder.
	outputs = decoder(features, captions)

	print('type(outputs):', type(outputs))
	print('outputs.shape:', outputs.shape)

	# Check that your decoder satisfies some requirements of the project! :D
	assert type(outputs)==torch.Tensor, "Decoder output needs to be a PyTorch Tensor."
	assert (outputs.shape[0]==batch_size) & (outputs.shape[1]==captions.shape[1]) & (outputs.shape[2]==vocab_size), "The shape of the decoder output is incorrect."

	return outputs


images, captions = get_sample_batch(data_loader)

features = check_encoder(images)
outputs = check_decoder(features, captions)

# if __name__ == "__main__":
# 	main()