import dataset
import train 
import network 
from datetime import datetime
import os 
import torch

def main():
    cur_time = str(datetime.now())
    summary_path = "./logs/" + cur_time
    model_path = "./models/" + cur_time
    os.mkdir(model_path)
    n_eval = 100
    hyperparameters = {"epochs": 10, "batch_size": 16}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # triplet_dataset = dataset.WhaleDataset("humpback-whale-identification/triplets.csv", "humpback-whale-identification/train/") 
    all_data = dataset.PairsDataset("humpback-whale-identification/randompairs.csv", "humpback-whale-identification/train/") 
    train_set, val_set = torch.utils.data.dataset.random_split(all_data, [int(len(all_data) * 0.8), len(all_data) - int(len(all_data) * 0.8)])
    model = network.SiameseNetwork()
    train.starting_train(
        val_set,
        train_set,
        model,
        hyperparameters,
        n_eval,
        summary_path,
        model_path,
        device,
    )

main()