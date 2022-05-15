import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from data_augment import RandomAugment
from dataset_generic import CustomPairDataset, load_dataframe
from model_zoo import SimCLRModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import set_device_and_seed


def train_simclr(net, data_loader, train_optimizer, device, temperature, epoch, epochs):
    '''
    Adapted from https://github.com/leftthomas/SimCLR
    '''
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(enumerate(data_loader))
    for i, (pos_1, pos_2, target) in train_bar:
        batch_size = pos_1.shape[0]*pos_1.shape[1]
        pos_1, pos_2 = pos_1.to(device), pos_2.to(device)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()
        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Batch: [{}/{}] Batch Loss: {:.4f}'.format(epoch, epochs, i, len(data_loader), loss.item()))
        
    return total_loss / total_num




def run_pretrain(run_wandb=True):
    with wandb.init() as run:
        configuration = wandb.config
        print("config for run", configuration)
        # set random seed and device CPU / GPU

        # info to load the data

        num_workers =4
        pin_memory=True

        tiles_summary_data = "/data/MSc_students_accounts/sneha/tiles_summary_5x_2.5mm_50mm_adapt_patch_size.csv"
        label_map = {"1a": 0, 
                         "1b": 0, 
                         "1c": 0, 
                         "2": 0,
                         "0": 0,
                         "3":1, 
                         "4":1,
                        "1": 0}
        
        tiles_summary = load_dataframe(tiles_summary_data, label_map)
        data_dir = '/data/goldin/images_raw/nafld_liver_biopsies_sirius_red/'
        base_dir = "/data/MSc_students_accounts/sneha/"

        device = set_device_and_seed(GPU=True, seed=configuration["seed"],
                                     gpu_name=configuration["gpu_name"])
        
        img_size = configuration["img_size"]
        batch_size = configuration["batch_size"]
        learning_rate = configuration["learning_rate"]
        weight_decay = configuration["weight_decay"]
        
        
        # get training transforms
        train_transform = RandomAugment(crop_scale= (configuration["min_crop"], 1.0), max_rotation=configuration["max_rotation"], kernel_size=configuration["kernel_size"],color=configuration["color_jitter"])

        
        print("Using train_transforms: ", train_transform)

        # build dataset and dataloader using WHOLE dataset for pre-training
        train_dataset = CustomPairDataset(tiles_summary, data_dir, use_red=configuration["use_red"], bag_size=configuration["bag_size"], transform=train_transform, img_size= img_size, verbose=False)

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=configuration["batch_size"], num_workers=num_workers, pin_memory=pin_memory)
        print("Created paired dataset and dataloader with batch size ", configuration["batch_size"])


        # load the model (ResNET18 + projection head)
        model = SimCLRModel(encoder=configuration["encoder"])
        print(model)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total number of parameters is: {}".format(params))
        use_all_gpus = configuration["multi_gpu"]
        if use_all_gpus:
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs! Model wrapped in DataParallel")
                model = nn.DataParallel(model)
                
        else: 
            print("Using 1 GPU only: ", device)

        model.to(device)
        
        if configuration["optimizer"] == "Adam":
            train_optimiser = optim.Adam(model.parameters(), lr=configuration["learning_rate"], weight_decay = configuration["weight_decay"])
        
        temperature = configuration["temperature"]
        results = {'train_loss': []}
        save_every = 100
#         set up 
        model_dir = base_dir + "test_files/models_simclr"
        os.makedirs(os.path.abspath(model_dir), exist_ok=True)
        os.chdir(model_dir)
        print(f"Starting model trainig for {configuration['num_epochs']} epochs")
        for epoch in range(configuration["num_epochs"]+1):
            train_loss = train_simclr(model.to(device), train_loader, train_optimiser, device, temperature, epoch, configuration["num_epochs"])
            results['train_loss'].append(train_loss)
            if run_wandb:
                wandb.log({"train_loss": train_loss})
            if epoch % configuration["save_every"] ==0:
                model.eval()
                torch.save(model.state_dict(), f"SimCLRmodel_epoch_{epoch}.h5")
                wandb.save(f"SimCLRmodel_epoch_{epoch}.h5")


        



if __name__ == "__main__":
    
    wandb.login()
    
    
    

    
    

    my_sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'train_loss',
        'goal': 'minimise'
    },
    'parameters': {

        'num_epochs': {
            'values': [100000]
        },

        'batch_size': {
            'values': [2]
        },
        'learning_rate': {
            'values': [1e-4]
        },
         'weight_decay': {
            'values': [1e-6]
        },
         'optimizer': {
            'values': ["Adam"]
        },
        'temperature': {
            'values': [0.1]
        },
        'img_size': {
            'value': 224
        },

         'color_jitter':  {
            'value': "light"
        },

        'kernel_size':  {
            'value': 0.1
        },


         'max_rotation': {
            'value': 270
        },

         'min_crop':  {
            'value': 0.01
        },

         'encoder': {
            'values': ["resnet18"]
        },

        'bag_size': {
           'value': 8
        },

        'seed': {
            'values': [42]
        },

        'use_red': {
            'value': False
        },

        'multi_gpu': {
            'value': False
        },

        'gpu_name': {
            'value': "cuda:0"
        },


        'save_every': {
            'value': 500
        }


        }
    }
    print(my_sweep_config)
    sweep_id = wandb.sweep(sweep=my_sweep_config, project="unsupervised_pretraining_2")
    wandb.agent(sweep_id, function=run_pretrain, count=1)


