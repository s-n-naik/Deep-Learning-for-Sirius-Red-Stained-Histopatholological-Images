import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from torchvision import transforms
from torchvision.utils import make_grid
from collections import OrderedDict

        
EPSILON = 0.00001
PIXEL_MIN_VALUE = 1.0/255.0


class StainStdCapsule(nn.Module):
    # Code adapted from https://github.com/Zhengyushan/ssc
    def __init__(self, routing_iter=3, stain_type=2, group_num=5, group_width=3, use_od=True, in_channels=3):
        '''
        routing_iter: # R number of dynamic routing iterations per forward pass
        stain_type: # S number of stain channels in normalised output 
        group_num: # M number of parallel convolutional groups
        group_width: # N width of each convolutional group
        use_od: True if optical density transform applied to input image, False if not
        in_channels: # channels in input image (3 for RGB, 4 for CMYK)
        '''
        super(StainStdCapsule, self).__init__()
        self.group_num = group_num
        self.stain_type = stain_type
        self.routing_iter = routing_iter
        self.width = group_width
        self.in_channels = in_channels
        self.use_od = use_od

        self.stain_presep = nn.Sequential(OrderedDict([
            ('ssc_conv0', nn.Conv2d(self.in_channels, group_num * group_width
                                    , kernel_size=1, bias=True, padding=0)),
            ('ssc_act0', nn.LeakyReLU()),
        ]))
        self.projection = nn.Sequential(OrderedDict([
            ('ssc_conv1', nn.Conv2d(group_num * group_width, group_num * stain_type,
                                    kernel_size=1, bias=False, padding=0, groups=group_num)),
            ('ssc_act1', nn.LeakyReLU()),
        ]))
        self.reconstruction = nn.Sequential(OrderedDict([
            ('ssc_conv_re', nn.Conv2d(stain_type, self.in_channels
                                      , kernel_size=1, bias=True, padding=0)),
            ('ssc_bn_re', nn.BatchNorm2d(self.in_channels)),
            ('ssc_act_re', nn.LeakyReLU()),
        ]))

    def forward(self, input_tensor):
        if self.use_od:
            od_input = -torch.log((input_tensor + PIXEL_MIN_VALUE))
        else:
            od_input = input_tensor
        x = self.stain_presep(od_input)
        x = self.projection(x)
        x = x.reshape(x.size(0), self.group_num, self.stain_type, x.size(2), x.size(3))
        if self.stain_type > 1:
            # Max 'Sparse Score'
            c = self.sparsity_routing(x)
        else:
            # Max Total Variation
            c = self.single_sparsity_routing(x)

        output = torch.sum(x * c, dim=1)
        re_image = self.reconstruction(output)
        if self.use_od:
            re_image = torch.exp(-re_image)

        return output, re_image

    def sparsity_routing(self, input_tensor):
        '''
        Modified dynamic routing algorithm for > 1 channel in normalised output
        Score to maximise = Sparse Score (maximum channel + pixel sparsity)
        
        '''
        u = input_tensor.data
        s = u
        b = 0.0
        for _ in range(self.routing_iter - 1):
            b = b + self.pixel_sparsity(s) + self.channel_sparsity(s)
            c = b.softmax(dim=1)
            s = torch.sum(c * u, dim=1, keepdim=True)
            s = s + u

        score = self.pixel_sparsity(s) + self.channel_sparsity(s)
        b = b + score
        c = b.softmax(dim=1)
        return c

    def single_sparsity_routing(self, input_tensor):
        '''
        Modified dynamic routing algorithm for 1 channel in normalised output
        Score to maximise = Total Variation of normalised output
        
        '''
        u = input_tensor.data
        s = u
        b = 0.0
        for _ in range(self.routing_iter - 1):
            b = b + self.single_stain_score(s)

            c = b.softmax(dim=1)
            s = torch.sum(c * u, dim=1, keepdim=True)
            s = s + u
        score = self.single_stain_score(s)
        b = b + score
        c = b.softmax(dim=1)
        return c

    def _tensor_size(self, input):
        '''
        Helper function for TV calculation
        
        '''
        return input.shape[2] * input.shape[3] * input.shape[4]

    def single_stain_score(self, x):
        '''
        Total Variation Score
        
        '''
        h_x = x.size()[3]
        w_x = x.size()[4]
        count_h = self._tensor_size(x[:, :, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, :, 1:])
        # add epsilon for stability of gradient images
        horiz_var = (x[:, :, :, 1:, :] - x[:, :, :, :h_x - 1, :]) + EPSILON
        vert_var = (x[:, :, :, :, 1:] - x[:, :, :, :, :w_x - 1]) + EPSILON
        # take l1 norm of gradient image
        l1_h = horiz_var.abs().sum(dim=2, keepdim=True).mean(dim=(3, 4), keepdim=True)
        l1_v = vert_var.abs().sum(dim=2, keepdim=True).mean(dim=(3, 4), keepdim=True)
        tv_score = l1_h + l1_v
        # upweight paths which have a higher total variation 
        return tv_score

    def pixel_sparsity(self, group_stains):
        '''
        Ensures not all pixels assigned to same channel
        
        '''
        values = group_stains + EPSILON
        l2 = values.pow(2).sum(dim=2, keepdim=True)
        l2 = l2.sqrt() + EPSILON
        l1 = values.abs().sum(dim=2, keepdim=True)
        sqrt_n = self.stain_type ** 0.5
        sparsity = (sqrt_n - l1 / l2) / (sqrt_n - 1)
        sparsity = sparsity.mean(dim=(3, 4), keepdim=True)
        return sparsity

    def channel_sparsity(self, group_stains):
        '''
        Ensures one pixel assigned to only one channel
        
        '''
        values = group_stains + EPSILON
        l2 = values.pow(2).sum(dim=(3, 4), keepdim=True)
        l2 = l2.sqrt() + EPSILON
        l1 = values.abs().sum(dim=(3, 4), keepdim=True)
        sqrt_n = (group_stains.size(3) * group_stains.size(4)) ** 0.5
        sparsity = (sqrt_n - l1 / l2) / (sqrt_n - 1)
        sparsity = sparsity.mean(dim=2, keepdim=True)
        return sparsity



class SSCMaxPoolClassifier(nn.Module):
    def __init__(self, encoder, output_features=1, num_layers = 1 , freeze=False, dropout=None, ssc_args=None):
        super(SSCMaxPoolClassifier, self).__init__()
        '''
        encoder: instance of class Encoder() - resnet / se-resnet 18/34 feature extractor
        output_features: number of classes for classification task (= 1 for binary classification)
        num_layers: number of linear layers before max-pool aggregation
        freeze: "all / True", "part", "none / False" to freeze weights in encoder
        dropout: True, then add drop out between linear layers
        scc_args: # R, # S, # M, # N, use_od, in_channels for StainStdCapsule() input params
        
        '''
        if ssc_args is not None:
            print("=> creating ssc module")
            self.ssc_module = StainStdCapsule(
                    routing_iter=ssc_args["num_routings"],
                    stain_type=ssc_args["num_stains"],
                    group_num=ssc_args["num_groups"], 
                    group_width=ssc_args["group_width"],
                    use_od = ssc_args["use_od"],
                    in_channels = ssc_args["in_channels"]
                )
        
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
        
        print(f"=> Ammending first conv layer of encoder to in_channels = {self.ssc_module.stain_type}")
        module_list = []
        for name, module in encoder.f.named_children():
            if "0" in name:
                module_list.append(torch.nn.Conv2d(self.ssc_module.stain_type, 64, kernel_size=7, stride=2,
                                padding=3, bias=False))
            else:
                module_list.append(module)
        self.f = nn.Sequential(*module_list)
        
        layers = []
        layers_size = [512, 256, 128, 64, 32, 16, 8, 4]
        assert num_layers < len(layers_size), f"Number of layers must be less than {len(layers_size)}"
        if num_layers == 1:
            layers.append(nn.Linear(512, output_features, bias=True))
        else:
            for i in range(num_layers-1):
                insize = layers_size[i]
                outsize = layers_size[i+1]
                layers.extend([nn.Linear(insize, outsize, bias=True), nn.ReLU()])
                if dropout is not None:
                    layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(outsize, output_features, bias=True))

        self.fc = nn.Sequential(*layers)
        
        
        
    def forward(self, inputs):
        batch_size = inputs.shape[0]
        bag_size = inputs.shape[1]
        inputs = inputs.reshape(bag_size*batch_size, inputs.shape[2], inputs.shape[3], inputs.shape[4])
        normed_input, reconst = self.ssc_module(inputs)
        x = self.f(normed_input)
        x = torch.flatten(x, start_dim=1)
        h = self.fc(x)
        preds = torch.sigmoid(h)
        preds = preds.reshape(batch_size, bag_size)
        output = torch.max(preds,1, keepdims=True)
        output_values = output.values
        max_bag_indices = output.indices
        output_values = output_values.squeeze(1)
        max_bag_indices = max_bag_indices.squeeze(1)
        
        return output_values, reconst, normed_input
    
    

class SSCGatedAttentionClassifier(nn.Module):
    def __init__(self, encoder, L=512, D=256, dropout=False, freeze=True, n_classes=1, ssc_args=None):
        super(SSCGatedAttentionClassifier, self).__init__()
        '''
        encoder: instance of class Encoder() - resnet / se-resnet 18/34 feature extractor
        L: encoder output feature dim
        D: linear layer output dim
        dropout: True, then add drop out between attention layers
        freeze: "all / True", "part", "none / False" to freeze weights in encoder
        n_classes: # classes for classification task (= 1 for binary)
        scc_args: # R, # S, # M, # N, use_od, in_channels for StainStdCapsule() input params    
        '''
        if ssc_args is not None:
            print("=> creating ssc module")
            self.ssc_module = StainStdCapsule(
                routing_iter=ssc_args["num_routings"],
                stain_type=ssc_args["num_stains"],
                group_num=ssc_args["num_groups"],
                group_width=ssc_args["group_width"],
                use_od=ssc_args["use_od"],
                in_channels=ssc_args["in_channels"]
            )

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

        print(f"=> Ammending first conv layer of encoder to in_channels = {self.ssc_module.stain_type}")
        module_list = []
        for name, module in encoder.f.named_children():
            if "0" in name:
                module_list.append(torch.nn.Conv2d(self.ssc_module.stain_type, 64, kernel_size=7, stride=2,
                                                   padding=3, bias=False))
            else:
                module_list.append(module)
        self.f = nn.Sequential(*module_list)
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

    def forward(self, inputs, return_attn=False):
        batch_size = inputs.shape[0]
        bag_size = inputs.shape[1]
        inputs = inputs.reshape(bag_size * batch_size, inputs.shape[2], inputs.shape[3], inputs.shape[4])
        normed_input, reconst = self.ssc_module(inputs)
        x = self.f(normed_input)
        x = x.view(batch_size, bag_size, -1)
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A) 
        A = A.reshape(batch_size, bag_size)
        A = F.softmax(A, dim=1).unsqueeze(1)  
        M = torch.matmul(A, x)
        if len(M.shape) > 2:
            M = M.squeeze()
        Y_prob = self.classifier(M)  
        Y_hat = torch.ge(Y_prob, 0.5).float()  
        if len(Y_prob.shape) > 1:
            Y_prob, Y_hat = Y_prob.squeeze(1), Y_hat.squeeze(1)
        if return_attn:
            return Y_prob, reconst, normed_input, A
        else:
            return Y_prob, reconst, normed_input


    
    
    
def reconstruction_loss(input, reconst, mode="l2"):
    '''
    Reconstruction loss function for SSC module training
    input: original image
    reconst: reconstructed SSC output
    mode: "l2", "l1", "none" to choose loss function to apply
    '''
    if mode == "l2":
        return torch.nn.functional.mse_loss(input, reconst)
    elif mode == "none":
        mse = torch.nn.functional.mse_loss(input, reconst)
        empty = torch.zeros_like(mse)
        return empty
    elif mode == "l1":
        return torch.nn.functional.l1_loss(input, reconst)
   
    


def train_model_w_ssc(model, optimizer, train_loader, device,criterion, num_epochs, count_epochs=0, reconst_mode="l2", verbose=True, scheduler=None, visualise=False, accum_iter=1):
    '''
    Training function for SSCGatedAttentionClassifier and SSCMaxPoolClassifier
    '''
    
    plotting_dict_train = {"train_loss_class":[], "accuracy": [], "train_loss_recon": [], "loss": []}
    for epoch in range(num_epochs):
        model.train()
        train_loss_class = 0
        train_loss_recon = 0
        train_error = 0
        correct = 0
        train_loss = 0
        denominator = 0
        for batch_idx, (data, label) in list(enumerate(train_loader)):
            bag_size = data.shape[1]
            denominator+= data.shape[0]
            data = data.to(device)
            bag_label = label.float().to(device)

            # gradient accumulation
            if batch_idx % accum_iter == 0:
                optimizer.zero_grad()
            # forward pass
            output_values, reconst, normed_input = model.forward(data)
            pred = output_values > 0.5
            pred = pred.long()
            correct += pred.eq(bag_label.view_as(pred)).sum().item()
            loss_class = criterion(output_values, bag_label)
            train_loss_class += loss_class.item()
            reshaped_data = data.view_as(reconst)
            loss_recon = reconstruction_loss(reshaped_data, reconst, mode=reconst_mode)
            train_loss_recon += loss_recon.item()
            
            loss = loss_class + loss_recon
            
            train_loss += loss.item()
            
            # backward pass
            loss = loss / accum_iter
            loss.backward()

            if (batch_idx % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.step()
                 
                
            if verbose > 0:
                print('Epoch: {}, Batch: {}, Bag size: {}, Loss: {:.2f}'.format(epoch+count_epochs, batch_idx, bag_size, loss.item()))
                
            if visualise:
                # visualise reconstructions 
                f, (ax1, ax2) = plt.subplots(1,2)
                img_list = [reconst[i] for i in range(reconst.shape[0])]
                img_grid = make_grid(img_list)
                npimg = img_grid.cpu().numpy()
                npimg = (npimg * 255).astype(np.uint8)
                ax1.imshow(np.transpose(npimg, (1,2,0)))
                ax1.set_title("Reconstructions")
                
                img_list = [reshaped_data[i] for i in range(reshaped_data.shape[0])]
                img_grid = make_grid(img_list)
                npimg = img_grid.cpu().numpy()
                
                ax2.imshow(np.transpose(npimg, (1,2,0)))
                ax2.set_title("Original Images")
                plt.savefig(f"reconstructions.jpg")
                plt.show()
                
        
        # calculate loss and error for epoch
        train_loss_class /= denominator
        train_loss_recon /= denominator
        train_loss /= denominator
        accuracy = correct / denominator
        plotting_dict_train["train_loss_class"].append(train_loss_class)
        plotting_dict_train["train_loss_recon"].append(train_loss_recon)
        plotting_dict_train["accuracy"].append(accuracy)
        plotting_dict_train["loss"].append(train_loss)
        
        # step scheduler at the end of each epoch
        if scheduler is not None:
            scheduler.step()
        
        print('Epoch: {}, Train Loss: {:.2f}, Train Accuracy: {:.2f}'.format(epoch+ count_epochs, train_loss, accuracy))
        
    return plotting_dict_train
        
def train_model_w_ssc_multiple_inference(model, optimizer, train_loader, device,criterion, num_epochs, count_epochs=0, reconst_mode="l2",  verbose=True, scheduler=None, visualise=False, accum_iter=1):
    '''
    Training function for SSCGatedAttentionClassifier and SSCMaxPoolClassifier with MultipleInferenceDataset
    
    '''
    plotting_dict_train = {"train_loss_class":[], "accuracy": [], "train_loss_recon": [], "loss": []}
    for epoch in range(num_epochs):
        model.train()
        train_loss_class = 0
        train_loss_recon = 0
        train_error = 0
        correct = 0
        train_loss = 0
        denominator = 0
        for batch_idx, (data, (label, path)) in enumerate(train_loader):
            bag_size = data.shape[1]
            denominator+= data.shape[0]
            data = data.to(device)
            bag_label = label.float().to(device)

            # reset gradients
            if batch_idx % accum_iter == 0:
                optimizer.zero_grad()

            # conduct a forward pass
            output_values, reconst, normed_input = model.forward(data)
            # calculate loss and metrics
            pred = output_values > 0.5
            pred = pred.long()
            correct += pred.eq(bag_label.view_as(pred)).sum().item()
            loss_class = criterion(output_values, bag_label)
            train_loss_class += loss_class.item()
            reshaped_data = data.view_as(reconst)
            loss_recon = reconstruction_loss(reshaped_data, reconst, mode=reconst_mode)
            train_loss_recon += loss_recon.item()
            
            loss = loss_class + loss_recon
            
            train_loss += loss.item()
            
            # backward pass
            loss = loss / accum_iter
            loss.backward()

            if (batch_idx % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.step()
                 
                
            if verbose > 0:
                print('Epoch: {}, Batch: {}, Bag size: {}, Loss: {:.2f}'.format(epoch+count_epochs, batch_idx, bag_size, loss.item()))
                
            if visualise:
                # visualise reconstructions 
                f, (ax1, ax2) = plt.subplots(1,2)
                img_list = [reconst[i] for i in range(reconst.shape[0])]
                img_grid = make_grid(img_list)
                npimg = img_grid.cpu().numpy()
                npimg = (npimg * 255).astype(np.uint8)
                ax1.imshow(np.transpose(npimg, (1,2,0)))
                ax1.set_title("Reconstructions")
                
                img_list = [reshaped_data[i] for i in range(reshaped_data.shape[0])]
                img_grid = make_grid(img_list)
                npimg = img_grid.cpu().numpy()
                
                ax2.imshow(np.transpose(npimg, (1,2,0)))
                ax2.set_title("Original Images")
                plt.savefig(f"reconstructions.jpg")
                plt.show()
   
        
        # calculate loss and error for epoch
        train_loss_class /= denominator
        train_loss_recon /= denominator
        train_loss /= denominator
        accuracy = correct / denominator
        plotting_dict_train["train_loss_class"].append(train_loss_class)
        plotting_dict_train["train_loss_recon"].append(train_loss_recon)
        plotting_dict_train["accuracy"].append(accuracy)
        plotting_dict_train["loss"].append(train_loss)
        
        # step at the end of each epoch
        if scheduler is not None:
            scheduler.step()
        
        print('Epoch: {}, Train Loss: {:.2f}, Train Accuracy: {:.2f}'.format(epoch+ count_epochs, train_loss, accuracy))
        
      
    return plotting_dict_train        
        

def get_scores_ssc_multiple_inference(model, test_loader, device, count_epochs, visualise=True, threshold=0.5):
    """
    Utility function to return a confusion matrix and metrics on test data.

    :param model: pytorch model
    :param test_loader: pytorch dataloader
    :param device: string 'cuda' or 'cpu'

    :return: cm, f1, accuracy: numpy array confusion matric, f1 score, accuracy score
    """
    print("Getting F1 score, Accuracy and Confusion matrix on the Multiple Inference Test Set")
    predictions = []
    y = []
    paths = []
    model.eval()
    with torch.no_grad():
        for i, (data, (label, path)) in enumerate(test_loader):
            data = data.to(device)
            label = label.to(device)
            path = torch.tensor(list(path))
            path = path.to(device)
            outputs, reconst, normed_input, A = model(data, return_attn=True)
            pred = outputs > threshold
            pred = pred.long()
            predictions.append(pred)
            y.append(label)
            paths.append(path)
            reshaped_data = data.view_as(reconst)
            if visualise:
                if i == 0:
                    # visualise reconstructions of first batch + save down
                    f, (ax1, ax2) = plt.subplots(2,1, figsize=(20,20))
                    img_list = [reconst[j] for j in range(reconst.shape[0])]
                    img_grid = make_grid(img_list)
                    npimg = img_grid.cpu().numpy()
                    ax1.imshow(np.transpose(npimg, (1,2,0)))
                    ax1.set_title("Reconstructions")

                    img_list = [reshaped_data[j] for j in range(reshaped_data.shape[0])]
                    img_grid = make_grid(img_list)
                    npimg = img_grid.cpu().numpy()
                    ax2.imshow(np.transpose(npimg, (1,2,0)))
                    ax2.set_title("Original Images")
                    plt.savefig(f"reconstructions_{count_epochs}_{i}.jpg")
                    plt.show()

                        
                        
                    # visualise outputs of first batch and save down
                    if normed_input.shape[1] > 1:
                        fig, axes = plt.subplots(normed_input.shape[1], 1, figsize=(20,20))
                        channel = 0

                        for row in axes:
                            img_list = [normed_input[j][channel].unsqueeze(0) for j in range(normed_input.shape[0])]
                            img_grid = make_grid(img_list)
                            npimg = img_grid.cpu().numpy()
                            row.imshow(np.transpose(npimg, (1,2,0)), cmap="gray")
                            row.set_title(f"Channel {channel}")
                            channel+=1
                        plt.savefig(f"normed_{count_epochs}_{i}.jpg")
                        plt.show()
                    else:
                        plt.figure()
                        img_list = [normed_input[j] for j in range(normed_input.shape[0])]
                        img_grid = make_grid(img_list)
                        npimg = img_grid.cpu().numpy()
                        plt.imshow(np.transpose(npimg, (1,2,0)))
                        plt.title(f"Single Channel")
                        plt.savefig(f"normed_{count_epochs}_{i}.jpg")
                        plt.show()
                    
                  
                    
    # get predictions per path - gather in dataframe and compute one-wins-all aggregation
    predictions = torch.cat(predictions, dim=0).cpu().numpy()
    y = torch.cat(y, dim=0).cpu().numpy()
    paths = torch.cat(paths, dim=0).cpu().numpy()
    predictions_df = pd.DataFrame({"prediction": predictions, "per_wsi_label": y, "wsi_path": paths}, columns=["prediction", "per_wsi_label", "wsi_path"])
    wsi_labels = []
    wsi_preds = []
    for wsi in np.unique(predictions_df["wsi_path"].tolist()):
        wsi_df = predictions_df[predictions_df["wsi_path"]==wsi]
        wsi_prediction = wsi_df["prediction"].max()
        assert len(np.unique(wsi_df["per_wsi_label"])) == 1, f"There is not a unique label for this WSI something has gone wrong:{np.unique(wsi_df['per_wsi_label'])} "
        wsi_label = wsi_df["per_wsi_label"].max()
        wsi_preds.append(wsi_prediction)
        wsi_labels.append(wsi_label)
    predictions_tensor = torch.tensor(wsi_preds).cpu().numpy()
    labels_tensor = torch.tensor(wsi_labels).cpu().numpy()
    
    cm = confusion_matrix(labels_tensor, predictions_tensor)
    f1 = f1_score(labels_tensor, predictions_tensor, average='binary')
    accuracy = accuracy_score(labels_tensor, predictions_tensor)
    print("F1 score for positive class (1): {:.2f}".format(f1))
    print("Accuracy: {:.2f}".format(accuracy))
    data = {'y_Actual':  labels_tensor.copy(),
        'y_Predicted': predictions_tensor.copy()
        }
    df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
    
    confusion_matrix_pd = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
    plt.figure()
    sn.heatmap(confusion_matrix_pd, annot=True)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.jpg")
    plt.show()
    return cm, f1, accuracy     



def get_scores_ssc(model, test_loader, device, count_epochs, visualise=True, threshold=0.5):
    """
    Utility function to return a confusion matrix and metrics on test data.

    :param model: pytorch model
    :param test_loader: pytorch dataloader
    :param device: string 'cuda' or 'cpu'

    :return: cm, f1, accuracy: numpy array confusion matric, f1 score, accuracy score
    """
    print("Getting F1 score, Accuracy and Confusion matrix on the Test Set")
    predictions = []
    y = []
    model.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.to(device)
            label = label.to(device)
            outputs, reconst, normed_input = model(data)
            pred = outputs > threshold
            pred = pred.long()
            predictions.append(pred)
            y.append(label)
            
            reshaped_data = data.view_as(reconst)
            
            if visualise:
                if i == 0:
                    # visualise reconstructions of first batch + save down
                    f, (ax1, ax2) = plt.subplots(2,1, figsize=(20,20))
                    img_list = [reconst[j] for j in range(reconst.shape[0])]
                    img_grid = make_grid(img_list)
                    npimg = img_grid.cpu().numpy()
                    ax1.imshow(np.transpose(npimg, (1,2,0)))
                    ax1.set_title("Reconstructions")

                    img_list = [reshaped_data[j] for j in range(reshaped_data.shape[0])]
                    img_grid = make_grid(img_list)
                    npimg = img_grid.cpu().numpy()
                    ax2.imshow(np.transpose(npimg, (1,2,0)))
                    ax2.set_title("Original Images")
                    plt.savefig(f"reconstructions_{count_epochs}_{i}.jpg")
                    plt.show()

                        
                    # visualise outputs of first batch and save down
                    if normed_input.shape[1] > 1:
                        fig, axes = plt.subplots(normed_input.shape[1], 1, figsize=(20,20))
                        channel = 0

                        for row in axes:
                            img_list = [normed_input[j][channel].unsqueeze(0) for j in range(normed_input.shape[0])]
                            img_grid = make_grid(img_list)
                            npimg = img_grid.cpu().numpy()
                            row.imshow(np.transpose(npimg, (1,2,0)), cmap="gray")
                            row.set_title(f"Channel {channel}")
                            channel+=1
                        plt.savefig(f"normed_{count_epochs}_{i}.jpg")
                        plt.show()
                    else:
                        plt.figure()
                        img_list = [normed_input[j] for j in range(normed_input.shape[0])]
                        img_grid = make_grid(img_list)
                        npimg = img_grid.cpu().numpy()
                        plt.imshow(np.transpose(npimg, (1,2,0)))
                        plt.title(f"Single Channel")
                        plt.savefig(f"normed_{count_epochs}_{i}.jpg")
                        plt.show()
                    

    predictions_tensor = torch.cat(predictions).cpu().numpy()
    labels_tensor = torch.cat(y).cpu().numpy()
    
    cm = confusion_matrix(labels_tensor, predictions_tensor)
    f1 = f1_score(labels_tensor, predictions_tensor, average='binary')
    accuracy = accuracy_score(labels_tensor, predictions_tensor)
    print("F1 score for positive class (1): {:.2f}".format(f1))
    print("Accuracy: {:.2f}".format(accuracy))
    data = {'y_Actual':  labels_tensor.copy(),
        'y_Predicted': predictions_tensor.copy()
        }
    df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
    confusion_matrix_pd = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
    plt.figure()
    sn.heatmap(confusion_matrix_pd, annot=True)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.jpg")
    plt.show()
    
    return cm, f1, accuracy  
