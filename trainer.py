from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
from loguru import logger
from tqdm import tqdm
import numpy as np
import logging
import random
import torch

logging.getLogger("transformers").setLevel(logging.ERROR)

class Trainer:
    def __init__(self,
        model_name: str,             # The name of the pre-trained model
        num_labels: int,             # The number of labels in the dataset
        epochs: int,                 # The number of epochs to train the model
        lr: float,                   # The learning rate of the optimizer
        output_dir: str,             # The output directory to save the model
        seed: int = None,            # The random seed for reproducibility
        eps: float = 1e-8,           # The epsilon value for AdamW optimizer
        metric: str = 'f1',          # The metric to evaluate the model
    ):
        logger.info('Start to initialize the Trainer')
        for k, v in locals().items():
            if k == 'self':
                continue
            logger.info(f'{k}: {v}')
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.epochs = epochs
        self.optimizer = AdamW(self.model.parameters(), lr=lr, eps=eps)
        self.output_path = f'{output_dir}/checkpoint-best-{metric}'
        self.device = Trainer.get_device()
        self.model.to(self.device)
        self.metric = metric
        if not seed:
            self.set_seed()

    @ staticmethod
    def get_device():
        ''' Returns the accessible device (CPU or GPU) '''
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f'There are {torch.cuda.device_count()} GPU(s) available.')
            logger.info(f'The current device is {torch.cuda.get_device_name(0)}.')
        else:
            device = torch.device('cpu')
            logger.info('No GPU available, using the CPU instead.')
        return device

    @ staticmethod
    def set_seed(seed_val=42):
        ''' Sets the random seed for reproducibility '''
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

    def metrics(self, logits, labels, n=4):
        '''
        Parameters:
            :logits: The predicted logits of the model
            :labels: The true labels of the data
            :n: The round precision to display

        Returns:
            metrics: A dictionary of metrics (accuracy, precision, recall, f1)
        '''
        pred_labels = np.argmax(logits, axis=1)
        acc = accuracy_score(labels, pred_labels)
        precision = precision_score(labels, pred_labels, average='weighted')
        recall = recall_score(labels, pred_labels, average='weighted')
        f1 = f1_score(labels, pred_labels, average='weighted')
        return {'accuracy': round(acc, n), 'precision': round(precision, n),'recall': round(recall, n), 'f1': round(f1, n)}

    def train(self, 
        train_dataloader: DataLoader,           # The DataLoader for training
        eval_dataloader: DataLoader=None        # The DataLoader for evaluation (optional)
    ) -> None:
        logger.info('Start to train the model')
        best_score = 0
        for epoch in range(self.epochs):
            all_logits, all_labels = [], []
            logger.info(f'Epoch {epoch+1}/{self.epochs}')
            
            self.model.train()
            train_loss = 0
            bar = tqdm(train_dataloader, total=len(train_dataloader))
            
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, labels, idx = batch
                
                self.optimizer.zero_grad()
                self.model.zero_grad()

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                logits = outputs[1]
                loss.backward()
                self.optimizer.step()
                
                all_logits.append(logits.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())

                train_loss += loss.item()
                bar.set_description(f'epoch {epoch + 1} loss: {train_loss/(step+1):.4f}')
                bar.update(1)
            bar.close()

            all_logits = np.concatenate(all_logits, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)

            logger.info(f'Training score')
            for k, v in self.metrics(all_logits, all_labels).items():
                logger.info(f'{k:<10}: {round(v*100, 4)}%')

            if eval_dataloader is not None:
                metrics = self.test(eval_dataloader)
                score = metrics[self.metric]
                if score > best_score:
                    best_score = score
                    logger.success(f'Saving the model to {self.output_path}')
                    self.model.save_pretrained(self.output_path)
                    
            else:
                logger.success(f'Saving the model to {self.output_path}')
                self.model.save_pretrained(self.output_path)

    def test(self, dataloader: DataLoader):
        self.model.eval()
        all_logits, all_labels = [], []
        for batch in tqdm(dataloader, total=len(dataloader), desc='Evaluating'):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, attention_mask, labels, idx = batch

            with torch.no_grad():
                output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = output[1]
            
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        result = self.metrics(all_logits, all_labels)
        logger.info(f'Evaluation score')
        for k, v in result.items():
            logger.info(f'{k:<10}: {round(v*100, 4)}%')
        return result
    
    def inference(self, dataloader: DataLoader):
        self.model.eval()
        all_logits = []
        for batch in tqdm(self.dataloader, total=len(self.dataloader), desc='Inference'):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, attention_mask, idx = batch

            with torch.no_grad():
                _, logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            all_logits.append(logits.detach().cpu().numpy())

        all_logits = np.concatenate(all_logits, axis=0)
        pred_labels = np.argmax(all_logits, axis=1)
        return pred_labels
