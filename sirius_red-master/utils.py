# Import
# import packages
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
from PIL import Image
from model_zoo import MaxPoolClassifier, Encoder, get_ozan_ciga, freeze_encoder, GatedAttention
from ssc_utils import SSCMaxPoolClassifier, train_model_w_ssc, SSCGatedAttentionClassifier, get_scores_ssc, \
    get_scores_ssc_multiple_inference, train_model_w_ssc_multiple_inference
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score


# set up


def get_model_ssc(configuration, device):
    '''
       Builds model according to specs in configuration
       1) get correct encoder [ simclr, resnet18/34, se_resnet18/34]
       2) if encoder has non imagenet pretrained weights, get these from the load_path provided for encoder
       3) get correct aggregator [SSCMaxPoolClassifier]
       4) Freeze / Part freeze encoder
       5) If multiple GPU trainnig, wrap model in DataParalell to facilitate

       return configured model

    '''
    if configuration["encoder"] == "ozan_ciga":
        encoder = get_ozan_ciga(device)
        encoder = freeze_encoder(encoder, freeze=configuration["freeze"])
    elif configuration["encoder"]=="simclr":
        encoder = Encoder(configuration["encoder"], img_net = configuration["image_net_pretrained"], freeze=configuration["freeze"])
    else:
        # Model  architecture - frst get encoder then aggregator
        encoder = Encoder(configuration["encoder"], img_net = configuration["image_net_pretrained"], freeze=configuration["freeze"])
    
    # load pre-trained model from direct file path
    if configuration["load_path"] is not False:
        weights = torch.load(os.path.abspath(configuration["load_path"]))
        encoder = load_model_weights(encoder, weights)

    if configuration["aggregation"] == "max_pool":
        ssc_args = {"num_routings": configuration["ssc_num_routings"],
            "num_stains": configuration["ssc_num_stains"],
            "num_groups": configuration["ssc_num_groups"],
            "group_width":configuration["ssc_group_width"],
                   "use_od": configuration["ssc_use_od"],
                   "in_channels": configuration["ssc_in_channels"]}
        
        model = SSCMaxPoolClassifier(encoder, num_layers = configuration["num_layers"], freeze=configuration["freeze"], dropout=configuration["dropout"], ssc_args=ssc_args)
        
    elif configuration["aggregation"] == "gated_attention":
        ssc_args = {"num_routings": configuration["ssc_num_routings"],
            "num_stains": configuration["ssc_num_stains"],
            "num_groups": configuration["ssc_num_groups"],
            "group_width":configuration["ssc_group_width"],
                   "use_od": configuration["ssc_use_od"],
                   "in_channels": configuration["ssc_in_channels"]}
        
        model = SSCGatedAttentionClassifier(encoder, freeze=configuration["freeze"], dropout=configuration["dropout"], ssc_args=ssc_args)

    if configuration["multi_gpu"]:
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs! Model wrapped in DataParallel")
            model = nn.DataParallel(model)

    else: 
        print("Using 1 GPU only: ", device)
        
    model.to(device)
    
    return model

def get_model(configuration, device):
    '''
       Builds model according to specs in configuration
       1) get correct encoder [ simclr, resnet18/34, se_resnet18/34]
       2) if encoder has non imagenet pretrained weights, get these from the load_path provided for encoder
       3) get correct aggregator [SimpleAttention, MaxPoolClassifier]
       4) Freeze / Part freeze encoder
       5) If multiple GPU trainnig, wrap model in DataParalell to facilitate

       return configured model

    '''
    if configuration["encoder"] == "ozan_ciga":
        encoder = get_ozan_ciga(device)
        encoder = freeze_encoder(encoder, freeze=configuration["freeze"])

    elif configuration["encoder"] == "simclr":
        encoder = Encoder(configuration["encoder"], img_net=configuration["image_net_pretrained"],
                          freeze=configuration["freeze"])
    else:
        # Model  architecture - frst get encoder then aggregator
        encoder = Encoder(configuration["encoder"], img_net=configuration["image_net_pretrained"],
                          freeze=configuration["freeze"])
    
    # load pre-trained model from direct file path
    if configuration["load_path"] is not False:
        weights = torch.load(os.path.abspath(configuration["load_path"]), map_location=configuration["gpu_name"])
        encoder = load_model_weights(encoder, weights)
        
    if configuration["aggregation"] == "max_pool":
        model = MaxPoolClassifier(encoder, num_layers=configuration["num_layers"], freeze=configuration["freeze"],
                                  dropout=configuration["dropout"])
    elif configuration["aggregation"] == "attention":
        model = SimpleAttention(encoder, freeze=configuration["freeze"])
    elif configuration["aggregation"] == "gated_attention":
        model = GatedAttention(encoder, freeze=configuration["freeze"], dropout=configuration["dropout"])

    if configuration["multi_gpu"]:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs! Model wrapped in DataParallel")
            model = nn.DataParallel(model)

    else:
        print("Using 1 GPU only: ", device)

    model.to(device)

    return model




def set_device_and_seed(GPU=True, seed=0, gpu_name = "cuda:0"):
    torch.cuda.is_available()
    if GPU:
        device = torch.device(gpu_name if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device("cpu")
    print(f'Using {device}')

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        
    set_seed(seed)
    
    return device

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Setting torch, numpy and random seeds to {seed}")


def show(img):
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))



def get_scores(model, test_loader, device, threshold=0.5):
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
        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)
            outputs, _ = model(data)
            pred = outputs > threshold
            pred = pred.long()
            predictions.append(pred)
            y.append(label)

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






def load_model_weights(model, weights):
    '''
    model = instance of model 
    weidghts = state_dict to be loaded (as many as possible)
    
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


def get_encoder(model):
    f = []
    for name, module in model.named_children():
        if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
            f.append(module)
    return nn.Sequential(*f)  


        
def get_scores_multiple_inference(model, test_loader, device, threshold=0.5):
    """
    Utility function to return a confusion matrix and metrics on test data.

    :param model: pytorch model
    :param test_loader: pytorch dataloader
    :param device: string 'cuda' or 'cpu'

    :return: cm, f1, accuracy: numpy array confusion matric, f1 score, accuracy score
    """
    print("Getting F1 score, Accuracy and Confusion matrix on the test set with multiple inference")
    predictions = []
    y = []
    paths = []
    model.eval()
    with torch.no_grad():
        for data, (label, path) in test_loader:
            data = data.to(device)
            label = label.to(device)
            path = torch.tensor(list(path))
            
            path = path.to(device)
            outputs, A = model(data, return_attn=True)
            pred = outputs > threshold
            pred = pred.long()
            predictions.append(pred)
            y.append(label)
            paths.append(path)
    predictions = torch.cat(predictions, dim=0).cpu().numpy()
    y = torch.cat(y, dim=0).cpu().numpy()
    paths = torch.cat(paths, dim=0).cpu().numpy()
    predictions_df = pd.DataFrame({"prediction": predictions, "per_wsi_label": y, "wsi_path": paths}, columns=["prediction", "per_wsi_label", "wsi_path"])
    wsi_labels = []
    wsi_preds = []
    for wsi in np.unique(predictions_df["wsi_path"].tolist()):
        wsi_df = predictions_df[predictions_df["wsi_path"]==wsi]
        wsi_prediction = wsi_df["prediction"].max()
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


if __name__ == "__main__":
    pass