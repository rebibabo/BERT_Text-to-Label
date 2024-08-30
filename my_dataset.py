from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer
from typing import List
from loguru import logger
from copy import deepcopy
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import pickle
import torch
import os

class MyDataset(Dataset):
    load: bool = False
    src_path: str = ''
    cache_path: str = ''

    def __init__(self, 
        src_path: str,              # source file path
        seed: bool = None,          # random seed for data split
        save: bool = True,          # whether to save the cache file
        *args, **kwargs             # other parameters for pandas.read_csv or pandas.read_json
    ):
        '''
        Three source file type: jsonl, csv, tsv
        jsonl:
            must have 'label' key in each line
        csv, tsv:
            if there is no head line,must include the parameter "name" that indicate the name of the label column.
            and make sure 'label' is in the column

        Example:
            dataset = MyDataset('a.jsonl', seed=42, save=False)
            dataset = MyDataset('b.csv', delimiter='\t', header=None, names=['label', 'code', 'filename', 'poison'])
        '''
        if seed is not None:
            self.set_seed(seed)
        self.src_path = src_path
        dirname = os.path.dirname(src_path)
        dirname = '.' if dirname == '' else dirname
        basename = '.'.join(os.path.basename(src_path).split('.')[:-1])
        self.cache_path = f'{dirname}/cache_{basename}'
        self.save = save
        if not os.path.exists(self.cache_path):
            logger.info(f'Loading source data from {src_path}')
            if src_path.endswith('.jsonl'):
                self.data = pd.read_json(src_path, lines=True)
            elif src_path.endswith('.csv') or src_path.endswith('.tsv'):
                self.data = pd.read_csv(src_path, *args, **kwargs)
            else:
                logger.critical('Unsupported file format.')
                return
        else:
            self.load = True
            logger.info(f'Loading cached data from {self.cache_path}')
            with open(self.cache_path, 'rb') as f:
                self.data = pickle.load(f)

    def subset(self, idx: List[int]):
        copy_self = deepcopy(self)
        copy_self.data = copy_self.data.iloc[idx]
        return copy_self

    def tokenize(self, 
        model_name: str,                # model name or path
        input_attr: str,                # attribute name of input text
        max_length: int = 64,           # maximum length of a sentence
        *args, **kwargs                 # other parameters for tokenizer
    ) -> None:
        if self.load:
            return
        tokenizer = AutoTokenizer.from_pretrained(model_name, *args, **kwargs)
        input_ids, attention_masks = [], []
        inputs = self.data[input_attr].tolist()
        for sentence in tqdm(inputs, desc='Tokenizing'):
            encoded_dict = tokenizer.encode_plus(
                sentence,                       # sentence to encode
                add_special_tokens = True,      # add [CLS] and [SEP]
                truncation=True,                # truncate sentence to max length
                padding='max_length',           # add padding
                max_length = max_length,        # maximum length of a sentence
                return_attention_mask = True,   # return attention mask
                return_tensors = 'pt',          # return PyTorch tensors
            )
            input_ids.append(encoded_dict['input_ids'][0])
            attention_masks.append(encoded_dict['attention_mask'][0])
        self.data['input_ids'] = input_ids
        self.data['attention_mask'] = attention_masks
        if not self.save:
            return
        logger.success(f'Saving cached data to {self.cache_path}')
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.data, f)

    def to_dataloader(self, 
        batch_size: int,            # batch size
        shuffle: bool = False       # whether to shuffle the data
    ) -> DataLoader:
        select_column = self.data[:][['input_ids', 'attention_mask', 'label']]
        select_tensor = []
        for name, column in select_column.to_dict().items():
            column_list = list(column.values())
            if isinstance(column_list[0], str):
                logger.warning('String type is not supported for DataLoader.')
                logger.warning(f'Deleting "{name}" attribute from DataLoader.')
            elif isinstance(column_list[0], int):
                select_tensor.append(torch.tensor(column_list))
            else:
                select_tensor.append(torch.stack(column_list))
        select_tensor.append(torch.tensor(range(len(self.data))))
        select_dataset = TensorDataset(*select_tensor)
        if shuffle:
            sampler = RandomSampler(select_dataset)
        else:
            sampler = SequentialSampler(select_dataset)
        return DataLoader(select_dataset, sampler=sampler, batch_size=batch_size)
    
    @ property
    def attr(self):
        return self.data.columns.tolist()

    @ staticmethod
    def set_seed(seed_val=42):
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, select_attr=None):
        return self.data.iloc[idx][select_attr] if select_attr else self.data.iloc[idx]

    def __str__(self):
        return f'MyDataset\n  source path\t\t{self.src_path}\n  attributes\t\t{self.attr}\n  sample number\t\t{len(self.data)}'

if __name__ == '__main__':
    dataset = MyDataset('./cola_public/raw/in_domain_train.tsv', delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
    dataset.tokenize('bert-base-uncased', 'sentence', do_lower_case=True)
    print(dataset)
    dataloader = dataset.to_dataloader(32, shuffle=True)
    for batch in dataloader:
        label, input_ids, attention_mask, idx = batch