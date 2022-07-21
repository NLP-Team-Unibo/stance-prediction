import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def train_loop(
        model, 
        optimizer, 
        criterion, 
        early_stopping, 
        loader_train, 
        loader_val, 
        epochs, 
        device,
        step_lr=None,
        cfg=''
    ):
    """
        This function excecute the full training procedure for a specified model. It logs all the results obtained during training using a SummaryWriter.

        Parameters
        ----------
        model: models.StancePredictionModule
            The stance prediction model to be trained.
        optimizer: optim.Optimizer
            The optimizer object.
        criterion: nn.Module
            The loss function of the classification task.
        early_stopping: utils.early_stopping.EarlyStopping
            The early stopping regularization object.
        loader_train: torch.utils.data.DataLoader
            The data loader of the training dataset.
        loader_val: torch.utils.data.DataLoader
            The data loader of the validation dataset.
        epochs: int
            The number of epochs to be excecuted.
        device: str
            The name of the device where to excecure the training procedure.
        step_lr: optim.Optimizer
            The learning rate scheduler. Default None.
        cfg: yacs.config.CfgNode or str
            The configuration file or a description containing all the (hyper)parameters of the model. Default ''.
            
    """
    # Push the model to the GPU if available
    if device == 'cuda':
        model.cuda()

    # Initialize the SummaryWriter
    writer = SummaryWriter('log/', flush_secs=1)
    writer.add_text('CFG', text_string=str(cfg))

    for i in range(epochs):
        # Execute a training step and store its results
        train_results = train(model, optimizer, criterion, loader_train, device)
        # Execute a validation step and store its results
        val_results = validate(model, criterion, loader_val, device)

        # Log the results
        writer.add_scalar('Train loss', train_results['train_loss'], i)
        writer.add_scalar('Train accuracy', train_results['train_accuracy'], i)
        writer.add_scalar('Val loss', val_results['val_loss'], i)
        writer.add_scalar('Val accuracy', val_results['val_accuracy'], i)

        # Check if the procedure must terminate due to early stopping
        if early_stopping:
            if early_stopping(val_results['val_accuracy']):
                print('Early stopping triggered, best score: ', early_stopping.best_score)
                model.load_state_dict(early_stopping.best_weights)
                break

        # Take a learining rate step
        if step_lr:
            step_lr.step()
        

def train(model, optimizer, criterion, data_loader, device):
    """
        This function excecute a single training step.

        Parameters
        ----------
        model: models.StancePredictionModule
            The stance prediction model to be trained.
        optimizer: optim.Optimizer
            The optimizer object.
        criterion: nn.Module
            The loss function of the classification task.
        data_loader: torch.utils.data.DataLoader
            The data loader of the training dataset.
        device: str
            The name of the device where to excecure the training procedure.

        Returns
        ----------
        results: dict
            A dictionary containing the loss and accuracy of this procedure.
    """
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_cls_loss = 0.0
    total_gen_loss = 0.0
    total = 0
    results = {}
    model_name = model.__class__.__name__
    for data in tqdm(data_loader):
        # Prepare the input according to the type of the model and propagate it to the network
        if model_name == 'TextModel':
            input_dict = data[0]
            input_dict = {k:input_dict[k].to(device) for k in input_dict.keys()}
            labels = data[1].to(device)
            output = model(**input_dict)
        elif model_name == 'AudioModel':
            waves = data[0].to(device)
            labels = data[1].to(device)
            output = model(waves)
        elif model_name == 'MultimodalModel':
            input_dict = data[0]
            input_dict = {k:input_dict[k].to(device) for k in input_dict.keys()}
            waves = data[1].to(device)
            labels = data[2].to(device)
            output = model(input_dict, waves)
        elif model_name == 'TextGenerationModel':
            input_dict = data[0]
            input_dict = {k:input_dict[k].to(device) for k in input_dict.keys()}
            waves = data[1].to(device)
            motion = data[2].to(device)
            labels = data[3].to(device)
            loss_lm, loss_cls, output = model(input_dict['input_ids'], input_dict['attention_mask'], waves, motion, labels_cls=labels, return_dict=False)

        # Compute the loss between the output and the target labels
        
        output = output.squeeze()
        if model_name != 'TextGenerationModel':
            loss = criterion(output, labels)
        else:
            total_cls_loss += loss_cls.item()
            total_gen_loss += loss_lm.item()
            loss = loss_lm + loss_cls

        total_loss += loss.item()
        acc = ((output > 0).float() == labels).sum().item()
        total_acc += acc
        total += labels.size(0)

        # Backpropagate and update the optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    #print('train_loss:', total_loss / len(data_loader), 'train_accuracy:', total_acc / total, end='\t')
    results['train_loss'] = total_loss / len(data_loader)
    results['train_accuracy'] = total_acc / total
    if model_name == 'TextGenerationModel':
        results['train_loss_gen'] = total_gen_loss / len(data_loader)
        results['train_loss_cls'] = total_cls_loss / len(data_loader)
    
    print('\t'.join([f'{key}: {results[key]}' for key in results.keys()]), end='\t')
    return results

def validate(model, criterion, data_loader, device):
    """
        This function excecute a single validation step.

        Parameters
        ----------
        model: models.StancePredictionModule
            The stance prediction model to be validated.
        criterion: nn.Module
            The loss function of the classification task.
        data_loader: torch.utils.data.DataLoader
            The data loader of the validation dataset.
        device: str
            The name of the device where to excecure the validation procedure.       

        Returns
        ----------
        results: dict
            A dictionary containing the loss and accuracy of this procedure.     
    """
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_acc = 0.0
        total_cls_loss = 0.0
        total_gen_loss = 0.0
        total = 0
        results = {}
        model_name = model.__class__.__name__
        for data in data_loader:
            # Prepare the input according to the type of the model and propagate it to the network
            if model_name == 'TextModel':
                input_dict = data[0]
                input_dict = {k:input_dict[k].to(device) for k in input_dict.keys()}
                labels = data[1].to(device)
                output = model(**input_dict)
            elif model_name == 'AudioModel':
                waves = data[0].to(device)
                labels = data[1].to(device)
                output = model(waves)
            elif model_name == 'MultimodalModel':
                input_dict = data[0]
                input_dict = {k:input_dict[k].to(device) for k in input_dict.keys()}
                waves = data[1].to(device)
                labels = data[2].to(device)
                output = model(input_dict, waves)
            elif model_name == 'TextGenerationModel':
                input_dict = data[0]
                input_dict = {k:input_dict[k].to(device) for k in input_dict.keys()}
                waves = data[1].to(device)
                motion = data[2].to(device)
                labels = data[3].to(device)
                loss_lm, loss_cls, output = model(input_dict['input_ids'], input_dict['attention_mask'], waves, motion, labels_cls=labels, return_dict=False)

            
            output = output.squeeze()
            # Compute the loss between the output and the target labels
            if model_name != 'TextGenerationModel':
                loss = criterion(output, labels)
            else:
                total_cls_loss += loss_cls.item()
                total_gen_loss += loss_lm.item()
                loss = loss_lm + loss_cls
            total_loss += loss.item()
            total += labels.size(0)

            acc = ((output > 0).float() == labels).sum().item()
            total_acc += acc
        results['val_loss'] = total_loss / len(data_loader)
        results['val_accuracy'] = total_acc / total
        if model_name == 'TextGenerationModel':
            results['val_loss_gen'] = total_gen_loss / len(data_loader)
            results['val_loss_cls'] = total_cls_loss / len(data_loader)
        print('\t'.join([f'{key}: {results[key]}' for key in results.keys()]))
        #print('val_loss:', total_loss / len(data_loader), 'val_accuracy:', total_acc / total)
    return results




"""metric = datasets.load_metric("rouge")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result"""