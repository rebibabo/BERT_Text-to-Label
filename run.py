from trainer import Trainer
from my_dataset import MyDataset

if __name__ == '__main__':
    names = ['sentence_source', 'label', 'label_notes', 'sentence']
    params = {"delimiter": "\t", "header": None, "names": names}

    train_dataset = MyDataset('./cola_public/raw/in_domain_train.tsv', **params)
    train_dataset.tokenize('bert-base-uncased', 'sentence', do_lower_case=True)
    train_dataloader = train_dataset.to_dataloader(16, shuffle=True)

    eval_dataset = MyDataset('./cola_public/raw/in_domain_dev.tsv', **params)
    eval_dataset.tokenize('bert-base-uncased', 'sentence', do_lower_case=True)
    eval_dataloader = eval_dataset.to_dataloader(16, shuffle=False)

    test_dataset = MyDataset('./cola_public/raw/out_of_domain_dev.tsv', **params)
    test_dataset.tokenize('bert-base-uncased', 'sentence', do_lower_case=True)
    test_dataloader = test_dataset.to_dataloader(16, shuffle=False)

    trainer = Trainer(
        model_name='bert-base-uncased',
        num_labels=2,
        epochs=4,
        lr=2e-5,
        output_dir='saved_models',
        seed=42,
        metric='f1'
    )
    trainer.train(train_dataloader=train_dataloader, eval_dataloader=eval_dataloader)
    trainer.test(test_dataloader)