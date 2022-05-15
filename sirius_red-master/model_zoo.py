# Import
import os
from typing import Any, Callable, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor
from torchvision.models import ResNet
from torchvision.models.utils import load_state_dict_from_url


#################FEATURE AGGREGATORS

class MaxPoolClassifier(nn.Module):
    '''
    Baseline architecture
    '''
    def __init__(self, encoder, output_features=1, num_layers = 1 , freeze=True, dropout=None):
        super(MaxPoolClassifier, self).__init__()
        
        # build model depending on num FC layers chosen + dropout
        layers = []
        layers_size = [512, 256, 128, 64, 32, 16, 8, 4]
        assert num_layers < len(layers_size), f"Number of layers must be less than {len(layers_size)}"
        if num_layers == 1:
            if dropout is not None:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(512, output_features, bias=True))
        else:
            for i in range(num_layers-1):
                insize = layers_size[i]
                outsize = layers_size[i+1]
                layers.extend([nn.Linear(insize, outsize, bias=True), nn.ReLU()])
            
            if dropout is not None:
                layers.append(nn.Dropout(p=dropout))
            
            # add final layer
            layers.append(nn.Linear(outsize, output_features, bias=True))

        self.fc = nn.Sequential(*layers)
        
        # freeze all layers if freeze=True, some layers if freeze=""
        for name, param in encoder.f.named_parameters():
            if freeze == "all" or freeze == True:
                param.requires_grad = False
            
            elif freeze == "part":
                if int(name[0]) < 6:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            elif freeze =="none"or freeze == False:
                param.requires_grad = True
        self.f = encoder
        
    def forward(self, inputs):
        batch_size = inputs.shape[0]
        bag_size = inputs.shape[1]
        inputs = inputs.reshape(bag_size*batch_size, inputs.shape[2], inputs.shape[3], inputs.shape[4])
        x = self.f(inputs)
        x = torch.flatten(x, start_dim=1)
        h = self.fc(x)
        preds = torch.sigmoid(h)
        # reshape
        preds = preds.reshape(batch_size, bag_size)
        output = torch.max(preds,1, keepdims=True)
        output_values = output.values
        max_bag_indices = output.indices
        
        output_values = output_values.squeeze(1)
        max_bag_indices = max_bag_indices.squeeze(1)
        
        return output_values, max_bag_indices




class GatedAttention(nn.Module):
    def __init__(self,encoder, L = 512, D = 256, dropout = False, freeze=True, n_classes = 1):
        super(GatedAttention, self).__init__()
        
        for name, param in encoder.f.named_parameters():
            if freeze == "all" or freeze == True:
                param.requires_grad = False
            elif freeze == "part":
                if int(name[0]) < 6:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            elif freeze =="none"or freeze == False:
                param.requires_grad = True
        
        self.f = encoder
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)
        
        self.classifier = nn.Sequential(
            nn.Linear(L, n_classes),
            nn.Sigmoid()
        )

    def forward(self, x, return_attn=True):
        # feature extracting using encoder
        batch_size = x.shape[0]
        bag_size = x.shape[1]
        x = x.reshape(batch_size * bag_size, x.shape[2], x.shape[3], x.shape[4])
        features = self.f(x)
        x = features.view(batch_size, bag_size, -1) # shape is (batch, bag, L)
        # attention based aggregation
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        A = A.reshape(batch_size, bag_size)
        A = F.softmax(A, dim=1).unsqueeze(1)  # softmax over bag features, unsqueeze for matmul dims
        M = torch.matmul(A, x)  # shape is (batch, L)
        if len(M.shape) > 2:
            M = M.squeeze()
        Y_prob = self.classifier(M) # unrounded sigmoid output shape is (batch, 1)
        Y_hat = torch.ge(Y_prob, 0.5).float() # round the sigmoid to get predicted label
        if len(Y_prob.shape) > 1:
            Y_prob, Y_hat = Y_prob.squeeze(1), Y_hat.squeeze(1)
        if return_attn:
            return Y_prob, A
        else:
            return Y_prob, features.view(-1,512)



############################# ENCODER


class Encoder(nn.Module):
    def __init__(self, model_name, img_net=True, freeze=True):
        '''
        Output is a 512dim vector

        Supports resnet18, resnet34, se-net18, se-net34,

        '''
        super(Encoder, self).__init__()

        self.img_net = img_net

        model_list = ["resnet18", "resnet34", "se_resnet34", "se_resnet18", "simclr", "densenet121"]

        assert model_name in model_list, f"Model name must be one of {model_list}"

        if model_name == "resnet18":
            model = models.resnet18(pretrained=self.img_net)
            feature_extractor = get_encoder(model)

        elif model_name == "resnet34":
            model = models.resnet34(pretrained=self.img_net)
            feature_extractor = get_encoder(model)

        elif model_name == "se_resnet34":
            model = se_resnet34(pretrained=self.img_net)
            feature_extractor = get_encoder(model)


        elif model_name == "se_resnet18":
            model = se_resnet18(pretrained=self.img_net)
            feature_extractor = get_encoder(model)

        elif model_name == "simclr":
            model = SimCLRModel().f
            feature_extractor = get_encoder(model)

        elif model_name == "densenet121":
            model = models.densenet121(pretrained=self.img_net)
            feature_extractor = get_encoder(model)

        self.f = feature_extractor

        for name, param in self.f.named_parameters():
            if freeze == "all" or freeze == True:
                param.requires_grad = False
            elif freeze == "part":
                if int(name[0]) < 6:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            elif freeze == "none" or freeze == False:
                param.requires_grad = True

    def forward(self, x):

        if len(x.shape) == 4:
            h = self.f(x)

        elif len(x.shape) == 5:
            batch_size = x.shape[0]
            bag_size = x.shape[1]
            h = self.f(x.reshape(batch_size * bag_size, x.shape[2], x.shape[3], x.shape[4]))

        else:
            raise TypeError(f"Input must be a 4 or 5d tensor not {x.shape}")

        return h.view(-1, 512)
    
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

class SEBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        r: int = 16
    ) -> None:
        super(SEBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        # SE block
        self.se = SE_Block(planes, r)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # SE operation
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def _resnet(
    arch: str,
    block,
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model = load_model_weights_2(model, state_dict)
    return model





def se_resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', SEBasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def se_resnet34(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet34', SEBasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)




################# GENERAL MODEL FUNCTIONS

def get_encoder(model):
    '''
    Removes last linear layer of model
    '''
    f = []
    for name, module in model.named_children():
        if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
            f.append(module)
    return nn.Sequential(*f)


def load_model_weights_2(model, weights):
    '''
    model = instance of model
    weights = state_dict to be loaded

    Function will load as many weights as it can match

    '''

    model_dict = model.state_dict()
    weights = {k.replace("module.", ""): v for k, v in weights.items() if k.replace("module.", "") in model_dict}
    keys_ = set(model_dict.keys())
    z = keys_.difference(set(weights.keys()))
    print(f"Found {len(weights)} out of {len(model_dict)} matching weights. \nNo matching weights for {z} ")
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    model.load_state_dict(model_dict)
    print("Successfully loaded model weights")
    return model


def freeze_encoder(encoder, freeze=True):
    '''
    Takes in an Encoder instance and freezes it all / part or not at all

    '''
    assert freeze in ["part", True, False, "all", "none"]

    for name, param in encoder.f.named_parameters():
        if freeze == "all" or freeze == True:
            param.requires_grad = False
        elif freeze == "part":
            if int(name[0]) < 6:
                param.requires_grad = False
            else:
                param.requires_grad = True
        elif freeze == "none" or freeze == False:
            param.requires_grad = True

    return encoder

######################SIMCLR MODELS

class SimCLRModel(nn.Module):
    '''
    Model for unsupervised pretraining using SimCLR

    adapted from https://github.com/leftthomas/SimCLR

    '''

    def __init__(self, feature_dim=128, encoder="resnet18"):
        super(SimCLRModel, self).__init__()

        self.f = []
        if encoder == "resnet18":
            encoder = models.resnet18()
        elif encoder == "se_resnet18":
            encoder = se_resnet18()

        for name, module in encoder.named_children():
            if name == 'conv1':
                # replaces Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(512, 256, bias=False), nn.BatchNorm1d(256),
                               nn.ReLU(inplace=True), nn.Linear(256, feature_dim, bias=True))

    def forward(self, x):
        batch_size = x.shape[0]
        bag_size = x.shape[1]
        x = x.reshape(batch_size * bag_size, x.shape[2], x.shape[3], x.shape[4])
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


def get_ozan_ciga(device, output_features=1,
                  MODEL_PATH="/data/MSc_students_accounts/sneha/test_files/self-supervised-histopathology-tenpercent/ozan_ciga.ckpt"):
    '''
    Function to load pretrained weights from https://github.com/ozanciga/self-supervised-histopathology
    returns: Resnet18 encoder with SimCLR pretrained weights

    '''
    MODEL_PATH = os.path.abspath(MODEL_PATH)
    model = models.__dict__['resnet18'](pretrained=False)

    state = torch.load(MODEL_PATH)  # , map_location='cuda:0'

    state_dict = state['state_dict']
    for key in list(state_dict.keys()):
        state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
    model_dict = model.state_dict()
    model = load_model_weights_2(model, state_dict)

    # ammending to remove fc layer
    print("Getting the encoder")
    f = []
    for name, module in model.named_children():
        if "fc" in name:
            print("Not adding fc layer")
        else:
            f.append(module)

    model = nn.Sequential(*f)
    encoder = Encoder("resnet18")
    encoder.f = model
    return encoder



if __name__ == "__main__":
    
    pass
