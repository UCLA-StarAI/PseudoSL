from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class SudokuLSTM(nn.Module):
    def __init__(self, boardSz, hidden_dim=128):
        super().__init__()

        self.boardSz = boardSz
        # self.lstm = nn.LSTM(batch_first=True, input_size=9, hidden_size=hidden_dim, num_layers=3)
        self.linear = nn.Linear(hidden_dim*2, 9)

    def __call__(self, x, is_inputs=None):
        # h0 = torch.randn(1, 100, 512).cuda()
        lstm_out, h = self.lstm(x)
        cell_space = self.linear(F.tanh(lstm_out))
        return cell_space, h#F.log_softmax(cell_space, dim=-1)


        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)

#https://github.com/SatyamGaba/image_captioning/blob/master/model.py
class CNNLSTM(nn.Module):
    def __init__(self, out_features, in_channels, arch_params, embed_size=128, hidden_size=512):
        super().__init__()

        self.model = torchvision.models.resnet18(pretrained=False, num_classes=out_features, **arch_params)

        # Hacking ResNets to expect 'in_channels' input channel (and not three)
        del self.model.conv1
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # map to the image features
        self.model.fc = nn.Linear(self.model.fc.in_features, embed_size)

        self.embed = nn.Embedding(3, embed_size)
        #self.embed = nn.Embedding(2, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size*2, hidden_size=hidden_size, num_layers=1, dropout=0.0, batch_first=True)
        #self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=2, dropout=0.7, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def __call__(self, x, y):
        embeddings = self.model(x)
        #embeddings = torch.cat((embeddings.unsqueeze(1), self.embed(y.long())), 1)
        if embeddings.size(0) == 10:
            embeddings = embeddings.repeat_interleave(528, dim=0)
            embeddings = torch.cat((embeddings.unsqueeze(1).expand(-1, 264, -1), self.embed(y.long())), 2)
        else:
            embeddings = torch.cat((embeddings.unsqueeze(1).expand(-1, 264, -1), self.embed(y.long())), 2)
        hidden, _ = self.lstm(embeddings)
        outputs = self.linear(hidden).squeeze()
        return outputs 

    def sample_stoch(self, x, states=None):
        """Generate captions for given image features using greedy search."""
        
        # Image embeddings
        features = self.model(x)

        sampled_ids = []
        #inputs = torch.cat(features, self.embed(torch.tensor(2, device='cuda')))
        inputs = torch.cat((features.unsqueeze(1), self.embed(torch.tensor(2, device='cuda')).unsqueeze(0).unsqueeze(0).expand(features.size(0), 1, 128)), -1)
        for i in range(264):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            predicted = torch.bernoulli(outputs.sigmoid()).squeeze(1).long() # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
            inputs = torch.cat((features.unsqueeze(1), inputs), -1)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

    def sample(self, x, states=None):
        """Generate captions for given image features using greedy search."""
        
        # Image embeddings
        features = self.model(x)

        sampled_ids = []
        #inputs = torch.cat(features, self.embed(torch.tensor(2, device='cuda')))
        inputs = torch.cat((features.unsqueeze(1), self.embed(torch.tensor(2, device='cuda')).unsqueeze(0).unsqueeze(0).expand(features.size(0), 1, 128)), -1)
        for i in range(264):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            predicted = outputs.sigmoid().round().squeeze().long()      # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
            inputs = torch.cat((features.unsqueeze(1), inputs), -1)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

    #def sample(self, x, states=None):
    #    """Generate captions for given image features using greedy search."""
    #    
    #    # Image embeddings
    #    features = self.model(x)

    #    sampled_ids = []
    #    inputs = features.unsqueeze(1)
    #    for i in range(264):
    #        hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
    #        outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
    #        predicted = outputs.sigmoid().round().squeeze().int()      # predicted: (batch_size)
    #        sampled_ids.append(predicted)
    #        inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
    #        inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
    #    sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
    #    return sampled_ids

class extract_tensor(nn.Module):
    def forward(self,x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        # Reshape shape (batch, hidden)
        return tensor#[:, -1, :]

class extract_tensor(nn.Module):
    def forward(self,x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        # Reshape shape (batch, hidden)
        return tensor#[:, -1, :]

def get_model(model_name, out_features, in_channels, arch_params):
    return CNNLSTM(out_features, in_channels, arch_params)
    preloaded_models = {"ResNet18": torchvision.models.resnet18}

    own_models = {"ConvNet": ConvNet, "MLP": MLP, "PureConvNet": PureConvNet, "CombResnet18": CombRenset18}

    if model_name in preloaded_models:
        model = preloaded_models[model_name](pretrained=False, num_classes=out_features, **arch_params)

        # Hacking ResNets to expect 'in_channels' input channel (and not three)
        del model.conv1
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.avgpool = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(512*3*3, 264*64)
        )
        model.fc = nn.Sequential(
            nn.Unflatten(1, (264, 64)),
            nn.LSTM(batch_first=True, input_size=64, hidden_size=64, num_layers=2, bidirectional=False, dropout=0.2),
            extract_tensor(),
            nn.Flatten(1),
            nn.Linear(64*1*264, 264)
        )
        return model
    elif model_name in own_models:
        return own_models[model_name](out_features=out_features, in_channels=in_channels, **arch_params)
    else:
        raise ValueError(f"Model name {model_name} not recognized!")


def dim_after_conv2D(input_dim, stride, kernel_size):
    return (input_dim - kernel_size + 2) // stride


class CombRenset18(nn.Module):

    def __init__(self, out_features, in_channels):
        super().__init__()
        self.resnet_model = torchvision.models.resnet18(pretrained=False, num_classes=out_features)
        del self.resnet_model.conv1
        self.resnet_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        output_shape = (int(sqrt(out_features)), int(sqrt(out_features)))
        self.pool = nn.AdaptiveMaxPool2d(output_shape)
        #self.last_conv = nn.Conv2d(128, 1, kernel_size=1,  stride=1)


    def forward(self, x):
        x = self.resnet_model.conv1(x)
        x = self.resnet_model.bn1(x)
        x = self.resnet_model.relu(x)
        x = self.resnet_model.maxpool(x)
        x = self.resnet_model.layer1(x)
        #x = self.resnet_model.layer2(x)
        #x = self.resnet_model.layer3(x)
        #x = self.last_conv(x)
        x = self.pool(x)
        x = x.mean(dim=1)
        return x


class ConvNet(torch.nn.Module):
    def __init__(self, out_features, in_channels, kernel_size, stride, linear_layer_size, channels_1, channels_2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=channels_1, kernel_size=kernel_size, stride=stride)
        self.conv2 = nn.Conv2d(in_channels=channels_1, out_channels=channels_2, kernel_size=kernel_size, stride=stride)

        output_shape = (4, 4)
        self.pool = nn.AdaptiveAvgPool2d(output_shape)

        self.fc1 = nn.Linear(in_features=output_shape[0] * output_shape[1] * channels_2, out_features=linear_layer_size)
        self.fc2 = nn.Linear(in_features=linear_layer_size, out_features=out_features)

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MLP(torch.nn.Module):
    def __init__(self, out_features, in_channels, hidden_layer_size):
        super().__init__()
        input_dim = in_channels * 40 * 20
        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_layer_size)
        self.fc2 = nn.Linear(in_features=hidden_layer_size, out_features=out_features)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


class PureConvNet(torch.nn.Module):

    act_funcs = {"relu": F.relu, "tanh": F.tanh, "identity": lambda x: x}

    def __init__(self, out_features, pooling, use_second_conv, kernel_size, in_channels,
                 channels_1=20, channels_2=20, act_func="relu"):
        super().__init__()
        self.use_second_conv = use_second_conv

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=channels_1, kernel_size=kernel_size, stride=1)
        self.conv2 = nn.Conv2d(in_channels=channels_1, out_channels=channels_2, kernel_size=kernel_size, stride=1)

        output_shape = (int(sqrt(out_features)), int(sqrt(out_features)))
        if pooling == "average":
            self.pool = nn.AdaptiveAvgPool2d(output_shape)
        elif pooling == "max":
            self.pool = nn.AdaptiveMaxPool2d(output_shape)

        self.conv3 = nn.Conv2d(in_channels=channels_2 if use_second_conv else channels_1,
                               out_channels=1, kernel_size=1, stride=1)
        self.act_func = PureConvNet.act_funcs[act_func]

    def forward(self, x):
        x = self.act_func(self.conv1(x))
        if self.use_second_conv:
            x = self.act_func(self.conv2(x))
        x = self.pool(x)
        x = self.conv3(x)
        return x
