# Fine-tuning the Bert on text-to-label task

There are two main class, MyDataset(my_dataset.py) and Trainer(trainer.py), the main training function is run.py

## myDataset
First, import the dataset package.
```python
from my_dataset import MyDataset
```
The source file must be in 'jsonl', 'csv', or 'tsv' format, each of which must contained a attribute called 'label', for example, a possible jsonl file may be:
```
{"filename": "c_24_2221.c", "label": "24", "code": "void main() {\n    char s[10000];\n    int i, j, m = 0, n = 0, l;\n    gets(s);\n    l = strlen(s);\n    for (i = 0; i <= l - 1; i++) {\n        if (((s[i] >= 'a') && (s[i] <= 'z')) ||\n            ((s[i] >= 'A') && (s[i] <= 'Z'))) {\n            for (j = i; j <= l - 1; j++) {\n                if (((s[j] >= 'a') && (s[j] <= 'z')) ||\n                    ((s[j] >= 'A') && (s[j] <= 'Z'))) {\n                    if (j != l - 1)\n                        continue;\n                    else {\n                        if ((j - i + 1) > (m - n)) {\n                            m = j + 1;\n                            n = i;\n                        }\n                        i = j;\n                        break;\n                    }\n                } else {\n                    if ((j - i) > (m - n)) {\n                        m = j;\n                        n = i;\n                    }\n                    i = j - 1;\n                    break;\n                }\n            }\n        }\n    }\n    for (i = n; i <= m - 1; i++)\n        printf(\"%c\", s[i]);\n    printf(\"\\n\");\n    m = 100;\n    n = 0;\n    for (i = 0; i <= l - 1; i++) {\n        if (((s[i] >= 'a') && (s[i] <= 'z')) ||\n            ((s[i] >= 'A') && (s[i] <= 'Z'))) {\n            for (j = i; j <= l - 1; j++) {\n                if (((s[j] >= 'a') && (s[j] <= 'z')) ||\n                    ((s[j] >= 'A') && (s[j] <= 'Z'))) {\n                    if (j != l - 1)\n                        continue;\n                    else {\n                        if ((j - i + 1) < (m - n)) {\n                            m = j + 1;\n                            n = i;\n                        }\n                        i = j;\n                        break;\n                    }\n                } else {\n                    if ((j - i) < (m - n)) {\n                        m = j;\n                        n = i;\n                    }\n                    i = j - 1;\n                    break;\n                }\n            }\n        }\n    }\n    for (i = n; i <= m - 1; i++)\n        printf(\"%c\", s[i]);\n    printf(\"\\n\");\n}\n"}
```
The loading code is as followed, if the parameter 'seed' is set to True, then the random shuffle will be duplicatable, if the parameter 'save' is set to True, then the tokenizing result will be saved to the same source path with the name starting with "cache", so there is no more worries about tokenizing over and over again. 
```python
dataset = MyDataset('a.jsonl', seed=42, save=False)
```

Given that the csv or tsv file may not contain a header, like the example below
```
gj04	0	*	They drank the pub.
```
Your should give a parameter "name" to give each column a name
```python
dataset = MyDataset('b.csv', delimiter='\t', header=None, names=['source', 'label', 'tag', 'sentence'])
```

After loading the source file, then run the tokenize method, inputting the tokenize's model_name, the attribute to be tokenized, and the max_length, then there will be two new attributes, input_ids and attention_mask
```python
train_dataset.tokenize('bert-base-uncased', 'sentence', do_lower_case=True)
```

Lastly, convert this dataset into the Dataloader, call the to_dataloader method, providing the batch_size and whether to shuffle, generally shuffle=True for training, and shuffle=False for Evaluating and testing.
```python
train_dataloader = train_dataset.to_dataloader(16, shuffle=True)
```

## Trainer
First, import the trainer package.
```python
from trainer import Trainer
```
To simply the training process, I construct a class Trainer, and the parameter's meaning is shown below
```python
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
```
You can load the model by:
```python
trainer = Trainer(
    model_name='bert-base-uncased',
    num_labels=2,
    epochs=4,
    lr=2e-5,
    output_dir='saved_models',
    seed=42,
    metric='f1'
)
```
Then train the model ,inputting the train_dataloader, if providing the eval_dataloader, each epoch will test in eval_dataloader
```python
trainer.train(train_dataloader=train_dataloader, eval_dataloader=eval_dataloader)
```
After training, it will save the best metric score to the output_dir, and call the test method:
```python
trainer.test(test_dataloader)
```
