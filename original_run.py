import torch
from loguru import logger
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from transformers import (BertTokenizer, BertForSequenceClassification, AdamW)
from torch.utils.data import (TensorDataset, random_split, 
                              DataLoader, RandomSampler, SequentialSampler)
import warnings
warnings.filterwarnings('ignore')
import random
import numpy as np

def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f'There are {torch.cuda.device_count()} GPU(s) available.')
        logger.info(f'The current device is {torch.cuda.get_device_name(0)}.')
    else:
        device = torch.device('cpu')
        logger.info('No GPU available, using the CPU instead.')
    return device

set_seed()
device = get_device()

df = pd.read_csv('./cola_public/raw/in_domain_train.tsv', delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
logger.info(f'The dataset has {len(df)} examples.')
sentences = df.sentence.values
labels = df.label.values

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

input_ids, attention_masks = [], []
for sentence in tqdm(sentences, desc='Tokenizing sentences', ncols=75):
    encoded_dict = tokenizer.encode_plus(
        sentence,                       # sentence to encode
        add_special_tokens = True,      # add [CLS] and [SEP]
        truncation=True,                # truncate sentence to max length
        padding='max_length',           # add padding
        max_length = 64,                # maximum length of a sentence
        return_attention_mask = True,   # return attention mask
        return_tensors = 'pt',          # return PyTorch tensors
    )
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

logger.info('Finished tokenizing the dataset.')
logger.info(f'The input_ids tensor has shape {input_ids.shape}.')

dataset = TensorDataset(input_ids, attention_masks, labels)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

logger.info(f'The training dataset has {len(train_dataset)} examples.')
logger.info(f'The validation dataset has {len(val_dataset)} examples.')

batch_size = 32
logger.info(f'The batch size is {batch_size}.')
train_dataloader = DataLoader(
            train_dataset,  
            sampler = RandomSampler(train_dataset), 
            batch_size = 32
        )

validation_dataloader = DataLoader(
            val_dataset, 
            sampler = SequentialSampler(val_dataset), 
            batch_size = 32
        )

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels = 2,
    output_attentions = False,
    output_hidden_states = False,
)

model.to(device)

optimizer = AdamW(model.parameters(),
                  lr = 2e-5,
                  eps = 1e-8            # tiny value to avoid division by zero
                )

epochs = 4

for epoch in range(epochs):
    logger.info(f'Epoch {epoch+1}/{epochs}')
    logger.info('-'*10)

    model.train()
    train_loss = 0
    correct = 0

    bar = tqdm(train_dataloader, total=len(train_dataloader), ncols=75)
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        optimizer.zero_grad()
        model.zero_grad()
        output = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)
        loss, logits = output[:2]
        loss.backward()
        optimizer.step()
        correct += logits.argmax(1).eq(b_labels).sum().item()
        train_loss += loss.item()

        bar.set_description(f'epoch {epoch + 1} loss: {train_loss/((step+1)*batch_size):.4f}')
        bar.update(1)
    bar.close()
    train_accuracy = correct / len(train_dataset)
    logger.info(f'Training accuracy: {train_accuracy:.4f}')
    model.eval()
    correct = 0
    for step, batch in enumerate(tqdm(validation_dataloader, total=len(validation_dataloader), ncols=75, desc='Validating')):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            output = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels)
            
        loss, logits = output[:2]
        correct += logits.argmax(1).eq(b_labels).sum().item()

    validate_accuracy = correct / len(val_dataset)
    logger.info(f'Validation accuracy: {validate_accuracy:.4f}')

df = pd.read_csv('./cola_public/raw/out_of_domain_dev.tsv', delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])

logger.info(f'The out-of-domain dataset has {len(df)} examples.')

sentences = df.sentence.values
labels = df.label.values


input_ids, attention_masks = [], []
for sentence in tqdm(sentences, desc='Tokenizing sentences', ncols=75):
    encoded_dict = tokenizer.encode_plus(
        sentence,                       # sentence to encode
        add_special_tokens = True,      # add [CLS] and [SEP]
        truncation=True,                # truncate sentence to max length
        padding='max_length',           # add padding
        max_length = 64,                # maximum length of a sentence
        return_attention_mask = True,   # return attention mask
        return_tensors = 'pt',          # return PyTorch tensors
    )
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])


input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)


dataset = TensorDataset(input_ids, attention_masks, labels)

test_dataloader = DataLoader(
            dataset, 
            sampler = SequentialSampler(dataset), 
            batch_size = 32
        )

model.eval()
correct = 0
for step, batch in enumerate(tqdm(test_dataloader, total=len(test_dataloader), ncols=75, desc='Testing')):
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():
        output = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)
        
    loss, logits = output[:2]
    correct += logits.argmax(1).eq(b_labels).sum().item()

test_accuracy = correct / len(dataset)
logger.info(f'Testing accuracy: {test_accuracy:.4f}')

# 保存模型
model_save_path = './bert_cola_model.pth'
torch.save(model.state_dict(), model_save_path)
logger.info(f'Model saved to {model_save_path}.')