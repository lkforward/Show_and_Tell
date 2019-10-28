import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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
        
        embed_captions = self.embed1(captions)
        # Expand the dim of features, so we have 3-dim as the inputs for LSTM:
        features = features.view(features.shape[0], 1, features.shape[1])

        # Combine image feature with the caption sequence, since image feautre is served as the first one in the seq.
        # Also permute the dimension so dim_0 is number of elements in the sequence, and dim_1 is number of batches. 
        inputs = torch.cat([features, embed_captions], dim=1).permute(1, 0, 2)
        print("The shape of lstm inputs:", inputs.shape)
        
        lstm_out, _ = self.lstm1(inputs)
        print("The shape of lstm output:", lstm_out.shape)
        # lstm_out has a shape (caption_length+1, n_batch, hidden_size)
        # tag_space has a shape (caption_length+1, n_batch, vocabulary)
        tag_space = self.hidden2tag(lstm_out)

        # The prediction of the sentence starts from the 2nd output:
        tag_space = tag_space[1:, :, :]
        # Put batch_size back to dim-0, and the new tag_space has a shape (n_batch, caption_length, vocabulary): 
        tag_space = tag_space.permute(1, 0, 2)
        
        # Apply the f(x) = log(softmax(x)) to the dim for a word: 
        tag_scores = F.log_softmax(tag_space, dim=-1)

        return tag_scores

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        # inputs is an embedd vector. 

        def sample_from_dist(tag_dist):
            """
            tag_dist: a 1d tensor of shape (vocab_size).
                The output from the forward calculation of the decoder. 
            """
            prob = torch.distributions.categorical.Categorical(tag_dist)
            word_token = prob.sample()    

            return word_token

        def forward_one_word(s_pre, h_pre, c_pre):
            """
            One step forward in the decoder LSTM. 
            [Input]: 
            s_pre: The word vector at previous step (t-1).
                If it is the image, it is the embedded vector with a shape of (embed_size);
                if it is a word, it is an integer corresponding to the word token.  
            [output]:
            s_t: is the predicted distribution at the current time step (t). 
            """

            # The \start token??? 
            start_token, end_token = 0, 1

            if s_pre.numel() == 1:
                # This is a word from the previous step
                embed_word = self.embed1(s_pre).view(1, 1, -1)
                o_t, (h_t, c_t) = self.lstm1(embed_word, (h_pre, c_pre))
                tag_space = self.hidden2tag(o_t)
                tag_scores = F.log_softmax(tag_space, dim=-1)

                dist = torch.squeeze(tag_scores)
                s_t = sample_from_dist(dist)
            else:
                # This is the input image:
                embed_input = s_pre.view(1, 1, -1)
                _, (h_t, c_t) = self.lstm1(embed_input, (h_pre, c_pre))
                s_t = start_token  # We only need h_t from the image.
            return s_t, (h_t, c_t)

        import pdb; pdb.set_trace()        
        if type(state)==tuple:
            assert len(state)==2, "The state should have both h_(t-1) and c_(t-1)"
            s_t, (h_t, c_t) = forward_one_word(inputs, state[0], state[1])
        else:
            # Feed the image vector into the LSTM (initializing the hidden as random)
            h_pre = torch.randn(1, 1, self.hidden_size)
            c_pre = torch.randn(1, 1, self.hidden_size)
            s_t, (h_t, c_t) = forward_one_word(inputs, h_pre, c_pre)

        # Feed each word into the LSTM
        prd_words = [start_token]
        while len(prd_words) < (max_len - 1):
            if s_t == end_token:
                prd_words.append(s_t)
            else: 
                s_t, (h_t, c_t) = forward_one_word(s_t, h_t, c_t)
                prd_words.append(s_t)

        # End of the prediction: 
        prd_words.append(end_token)

        return prd_words
