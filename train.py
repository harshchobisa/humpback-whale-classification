'''
---notes---
enumerate:
 - enumerate(parameter is an iterable object).
 - can be used to return an enumerate object
 - or if used in for loops can either store each tuple of index, element into one variable or separate them into two (what we do)
 - examples here: https://www.geeksforgeeks.org/enumerate-in-python/
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard
import tensorboard as tf 
import datetime 
import constants
import os

#have a parameter for "how often to log data to TensorBoard"
def starting_train(
    val_dataset,
    train_dataset,
    model,
    hyperparameters,
    n_eval,
    summary_path,
    model_path,
    device,
):
    """
    Trains and evaluates a model.

    Args:
        _dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
        summary_path:    Path where Tensorboard summaries are located.
    """

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders - creates a "list of batches" (sort of)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss() 

    # Initialize summary writer (for logging)
    writer = torch.utils.tensorboard.SummaryWriter(summary_path)

    # Move model + datasets to device
    model = model.to(device)
    # train_dataset = train_dataset.to(device)
    # val_dataset = val_dataset.to(device)

    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        for i, batch in enumerate(train_loader): #see notes above
            print(f"\rIteration {i + 1} of {len(train_dataset)} ...", end="")

            # Backpropagation and gradient descent
            image_one, image_two, labels = batch 
            images = torch.cat([image_one, image_two], dim=2) #pack images together for train efficiency. must unpack later
            output = model.forward(images) #do one call on each type of image concatenated together 

            #run the loss function on the embeddings
            loss = loss_fn(output, labels.long())
            accuracy = compute_accuracy(output, labels)
            loss.backward()
            optimizer.step()

            writer.add_scalar("train_loss", loss, global_step = step)
            writer.add_scalar("training_accuracy", accuracy, global_step = step)

            if step % n_eval == 0 and step != 0:
                accuracy, loss = evaluate(val_loader, model)
                writer.add_scalar("val_loss", loss, global_step = step)
                writer.add_scalar("validation_accuracy", accuracy, global_step = step)
                torch.save(model.state_dict(), os.path.join(model_path,'model.pt'))
            
            optimizer.zero_grad()
            step += 1

        print("Epoch " , epoch, " Loss ", loss)


def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """
    outputs = outputs.argmax(axis=1)

    n_correct = (outputs == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total

def evaluate(val_loader, model):
    """
    want to compare to a database of embeddings that we already have 
    """

    '''
    TODO save model and tensorboard every time evaluate happens 
    '''
    
    model.eval() #put network in eval mode
    with torch.no_grad():
        #calculate embeddings for the training set 
        for i, batch in enumerate(val_loader):

            image_one, image_two, labels = batch 
            images = torch.cat([image_one, image_two], dim=2) #pack images together for train efficiency. must unpack later
            output = model.forward(images) #do one call on each type of image concatenated together 

            #run the loss function on the embeddings
            loss_fn = nn.CrossEntropyLoss() 
            loss = loss_fn(output, labels.long())
            accuracy = compute_accuracy(output, labels)

    return accuracy, loss
