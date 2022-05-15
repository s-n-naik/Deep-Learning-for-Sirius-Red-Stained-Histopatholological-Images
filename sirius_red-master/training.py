# Import
# import packages
import torch.nn.functional as F


def train_model(model, optimizer, train_loader, device,criterion, num_epochs, count_epochs=0, verbose=True, scheduler=None, accum_iter=1):
    
    plotting_dict_train = {"loss":[], "accuracy": []}
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_error = 0
        correct = 0
        denominator = 0
        for batch_idx, (data, label) in list(enumerate(train_loader)):
            bag_size = data.shape[1]
            denominator+= data.shape[0]
            data = data.to(device)
            bag_label = label.float().to(device)

            # reset gradients
            if batch_idx % accum_iter ==0:
                optimizer.zero_grad()

            # conduct a forward pass
            output_values, max_indices = model.forward(data)

            # calculate loss and metrics
            pred = output_values > 0.5
            pred = pred.long()
            correct += pred.eq(bag_label.view_as(pred)).sum().item()
            
            loss = criterion(output_values, bag_label)
            
            train_loss += loss.item()

            # backward pass, normalising for gradient accumulation
            loss = loss / accum_iter
            loss.backward()
            
            # step
            if (batch_idx % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.step()

                
            if verbose:
                print('Epoch: {}, Batch: {}, Bag size: {}, Loss: {:.2f}'.format(epoch+count_epochs, batch_idx, bag_size, loss.item()))
        # calculate loss and error for epoch
        train_loss /= denominator
        accuracy = correct / denominator
        plotting_dict_train["loss"].append(train_loss)
        plotting_dict_train["accuracy"].append(accuracy)
        
        # step at the end of each epoch
        if scheduler is not None:
            scheduler.step()
            
            if scheduler is not None:
                last_lr = scheduler.get_last_lr()[0]
            
        print('Epoch: {}, Train Loss: {:.2f}, Train Accuracy: {:.2f}'.format(epoch+count_epochs, train_loss, accuracy))
        
    return plotting_dict_train
            

    
def train_model_multiple_inference(model, optimizer, train_loader, device,criterion, num_epochs, count_epochs=0, verbose=True, scheduler=None, accum_iter=1):
    
    plotting_dict_train = {"loss":[], "accuracy": []}
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_error = 0
        correct = 0
        denominator = 0
        for batch_idx, (data, (label, path)) in enumerate(train_loader):
            bag_size = data.shape[1]
            denominator+= data.shape[0]
            data = data.to(device)
            bag_label = label.float().to(device)
                
            # reset gradients
            if batch_idx % accum_iter ==0:
                optimizer.zero_grad()

            # conduct a forward pass
            output_values, max_indices = model.forward(data)

            # calculate loss and metrics
            pred = output_values > 0.5
            pred = pred.long()
            correct += pred.eq(bag_label.view_as(pred)).sum().item()
            loss = criterion(output_values, bag_label)
            train_loss += loss.item()

            # backward pass, normalising for gradient accumulation
            loss = loss / accum_iter
            loss.backward()
            
            # step
            if (batch_idx % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.step()

            if verbose:
                print('Epoch: {}, Batch: {}, Bag size: {}, Loss: {:.2f}'.format(epoch+count_epochs, batch_idx, bag_size, loss.item()))
        # calculate loss and error for epoch
        train_loss /= denominator
        accuracy = correct / denominator
        plotting_dict_train["loss"].append(train_loss)
        plotting_dict_train["accuracy"].append(accuracy)
        
        # step at the end of each epoch
        if scheduler is not None:
            scheduler.step()
       
            if scheduler is not None:
                last_lr = scheduler.get_last_lr()[0]
            
        print('Epoch: {}, Train Loss: {:.2f}, Train Accuracy: {:.2f}'.format(epoch+count_epochs, train_loss, accuracy))
        
    return plotting_dict_train      
    

    

if __name__ == "__main__":
    
    pass
