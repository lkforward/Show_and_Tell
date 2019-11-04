import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        # input_gate = nn.Linear(embed_size, hidden_size)
        super(DecoderRNN, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.embed1 = nn.Embedding(vocab_size, embed_size)
        self.lstm1 = nn.LSTM(input_size = embed_size, hidden_size = hidden_size, num_layers = num_layers)
        self.hidden2tag = nn.Linear(hidden_size, vocab_size)
        
        pass
    
    def forward(self, features, captions):
        """
        [INPUTS]:
        features：torch tensor, shape = (n_batch, embedding_size). 
            NOTE: This embedding is the embedding of the CNN output. 
        captions：torch tensor, shape = (n_batch, caption_length). 
            NOTE: We assume the captions all have the same length, the value in captions are integer indices. 
        """
        
        # -> (n_batch, caption_length - 1, embed_size)
        embed_captions = self.embed1(captions[:, :-1])

        # Expand the dim of features, so we have 3-dim as the inputs for LSTM:
        # -> (n_batch, 1, embedding_size).
        features = features.unsqueeze(1)
        
        # Combine image feature with the caption sequence, since image feautre is served as the first one in the seq.
        # Also permute the dimension so dim_0 is number of elements in the sequence, and dim_1 is number of batches. 
        # -> (caption_length, n_batch, embed_size)
        inputs = torch.cat([features, embed_captions], dim=1).permute(1, 0, 2)
        print("The shape of lstm inputs:", inputs.shape)
        
        # hidden = (torch.zeros((1, self.batch_size, self.hidden_size), device=device), 
        #         torch.zeros((1, self.batch_size, self.hidden_size), device=device))
        # lstm_out, _ = self.lstm1(inputs, hidden)
        lstm_out, _ = self.lstm1(inputs)
        print("The shape of lstm output:", lstm_out.shape)

        # (caption_length, n_batch, hidden_size) -> (caption_length, n_batch, vocabulary)
        tag_space = self.hidden2tag(lstm_out)

        # We have dropped the last element in the caption at the beginning, so no need to drop again here:
        # tag_space = tag_space[1:, :, :]
        # # Put batch_size back to dim-0, and the new tag_space has a shape (n_batch, caption_length, vocabulary): 
        tag_space = tag_space.permute(1, 0, 2)
        print("The shape of tag_space:", tag_space.shape)

        return tag_space

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        # inputs is an embedd vector. 

        start_token, end_token = 0, 1

        if (type(states) is tuple) and len(states)==2:
            hidden = states
        else: 
            h_pre = torch.randn(1, 1, self.hidden_size).to(inputs.device)
            c_pre = h_pre
            hidden = (h_pre, c_pre)
             
        prd_words = []
        while len(prd_words) < (max_len - 1):
            o_t, hidden = self.lstm1(inputs, hidden)
            tag_space = self.hidden2tag(o_t)  #  1,1,vocab_size
            tag_scores = tag_space.squeeze(1)  # 1,vocab_size

            word_ind = tag_scores.argmax(dim=-1)  # 1
            prd_words.append(word_ind.item()) 

            inputs = self.embed1(word_ind.unsqueeze(0)) # 1,1->1,1,embed_size

        # End of the prediction: 
        prd_words.append(end_token)

        return prd_words