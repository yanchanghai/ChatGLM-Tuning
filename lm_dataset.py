from datasets import Dataset, DatasetInfo, Features, ClassLabel
import torch
import numpy as np


# LMDataSet
class LMDataset(Dataset):
    def __init__(self, args):
        self.max_seq_length = args.max_seq_length
        self.data_path = args.data_path
        self.epoch_length = args.epoch_length
        if self.epoch_length < 1:
            self.epoch_length = 10000

        self.doc = np.memmap(self.data_path, dtype=np.int64, mode='r')

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, idx):
        i = np.random.randint(0, len(self.doc) - (self.max_seq_length + 1))  # cheat: pick a random spot in dataset
        chunk = self.doc[i:i + self.max_seq_length + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
