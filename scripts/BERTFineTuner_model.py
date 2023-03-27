
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW, SGD
import numpy as np

def accuracy_score(y_true, y_pred):
    """
    Calculates the accuracy score given true labels and predicted labels.
    """
    y_pred = np.round(y_pred).astype(int)
    accuracy = np.mean(y_true == y_pred)
    return accuracy
class BertTransferModel:
    """
    A class to fine-tune a BERT model for sentence pair classification tasks.

    Attributes
    ----------
    tokenizer : transformers.BertTokenizer
        BERT tokenizer for encoding sentences.
    model : transformers.BertForSequenceClassification
        BERT model for sequence classification.
    max_length : int
        Maximum length for encoding sentences (default: 128).
    batch_size : int
        Batch size for training (default: 16).
    learning_rate : float
        Learning rate for the optimizer (default: 2e-5).
    num_epochs : int
        Number of epochs for training (default: 3).

    Methods
    -------
    encode_sentences(sentences_1, sentences_2)
        Encodes two lists of sentences using the BERT tokenizer.
    create_dataset(sentences_1, sentences_2, labels)
        Creates a PyTorch dataset from the encoded sentences and labels.
    fit(train_sentences_1, train_sentences_2, train_labels, optimizer_name='AdamW')
        Fine-tunes the BERT model using the given sentences and labels.
    custom_dataloader_padder(batch):
        Pad to the good format for Dataloader object
    predict(sentences_1, sentences_2, model=None)
        Predicts the labels for the given sentence pairs using a pre-trained or trained model.
    """
    
    def __init__(self, model_name='bert-base-uncased', max_length=128, batch_size=16, learning_rate=2e-5, num_epochs=3):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
    def encode_sentences(self, sentences_1, sentences_2):
        """
        Encodes two lists of sentences using the BERT tokenizer.

        Parameters
        ----------
        sentences_1 : list of str
            The first set of sentences.
        sentences_2 : list of str
            The second set of sentences.

        Returns
        -------
        input_ids : torch.Tensor
            Tensor containing input IDs for each sentence pair.
        attention_masks : torch.Tensor
            Tensor containing attention masks for each sentence pair.
        token_type_ids : torch.Tensor
            Tensor containing token type IDs for each sentence pair.
        """

        input_ids, attention_masks, token_type_ids = [], [], []

        for sent1, sent2 in zip(sentences_1, sentences_2):
            encoded = self.tokenizer(sent1, sent2, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])
            token_type_ids.append(encoded['token_type_ids'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        token_type_ids = torch.cat(token_type_ids, dim=0)

        return input_ids, attention_masks, token_type_ids


    def create_dataset(self, sentences_1, sentences_2, labels):
        """
        Creates a PyTorch dataset from the encoded sentences and labels.

        Parameters
        ----------
        sentences_1 : list of str
            The first set of sentences.
        sentences_2 : list of str
            The second set of sentences.
        labels : list of int
            The corresponding labels for each sentence pair.

        Returns
        -------
        dataset : torch.utils.data.TensorDataset
            PyTorch dataset containing the input IDs, attention masks, token type IDs, and labels.
        """
        
        input_ids, attention_masks, token_type_ids = self.encode_sentences(sentences_1, sentences_2)
        labels = torch.tensor(labels, dtype=torch.long)
        dataset = TensorDataset(input_ids, attention_masks, token_type_ids, labels)
        return dataset
        
    def custom_dataloader_padder(self,batch):
        """
        Pads the input sequences to have the same length and returns a dataloader for the BERT model.
        """
        input_ids, attention_masks, token_type_ids, labels = zip(*batch)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
        labels = torch.tensor(labels, dtype=torch.long)
        return input_ids, attention_masks, token_type_ids, labels
    
    def fit(self, train_sentences_1, train_sentences_2, train_labels, val_sentences_1, val_sentences_2, val_labels, optimizer_name='AdamW'):
        """
        Fine-tunes the BERT model using the given sentences and labels.

        Parameters
        ----------
        train_sentences_1 : list of str
            The first set of training sentences.
        train_sentences_2 : list of str
            The second set of training sentences.
        train_labels : list of int
            The labels for the training data.
        val_sentences_1 : list of str
            The first set of validation sentences.
        val_sentences_2 : list of str
            The second set of validation sentences.
        val_labels : list of int
            The labels for the validation data.
        optimizer_name : str, optional
            The name of the optimizer to use for fine-tuning (default: 'AdamW').
            Supported optimizers: 'AdamW', 'SGD'.
        """

        train_dataset = self.create_dataset(train_sentences_1, train_sentences_2, train_labels)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.custom_dataloader_padder, drop_last=True)

        val_dataset = self.create_dataset(val_sentences_1, val_sentences_2, val_labels)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.custom_dataloader_padder, drop_last=False)

        optimizers = {
            'AdamW': AdamW(self.model.parameters(), lr=self.learning_rate),
            'SGD': SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        }

        optimizer = optimizers[optimizer_name]

        num_training_steps = len(train_dataloader) * self.num_epochs
        warmup_steps = int(0.1 * num_training_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

        train_accuracies, val_accuracies = [], []

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            epoch_preds, epoch_labels = [], []

            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs} (Train)"):
                optimizer.zero_grad()
                input_ids, attention_masks, token_type_ids, labels = batch
                outputs = self.model(input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids, labels=labels)
                loss = outputs.loss
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
                scheduler.step()

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                epoch_preds.extend(preds.detach().cpu().numpy())
                epoch_labels.extend(labels.detach().cpu().numpy())

            epoch_train_accuracy = accuracy_score(epoch_labels, epoch_preds)
            train_accuracies.append(epoch_train_accuracy)

            val_accuracy = self.evaluate(val_dataloader)
            val_accuracies.append(val_accuracy)

            print(f'Epoch: {epoch+1}, Train Loss: {running_loss/len(train_dataloader):.4f}, Train Accuracy: {epoch_train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}')

        return train_accuracies, val_accuracies
    
    def predict(self, sentences_1, sentences_2, model_path=None):
        """
        Predicts the labels for the given sentence pairs using the trained model.

        Parameters
        ----------
        sentences_1 : list of str
            The first set of sentences.
        sentences_2 : list of str
            The second set of sentences.
        model_path : str, optional
            The path to the saved model. If None, uses the trained model (default: None).

        Returns
        -------
        preds : list of int
            The predicted labels for each sentence pair.
        """
        if model_path:
            model = BertForSequenceClassification.from_pretrained(model_path)
        else:
            model = self.model

        input_ids, attention_masks, token_type_ids = self.encode_sentences(sentences_1, sentences_2)
        dataset = TensorDataset(input_ids, attention_masks, token_type_ids)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        model.eval()
        preds = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_masks, token_type_ids = batch
                outputs = model(input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)
                logits = outputs.logits
                batch_preds = torch.argmax(logits, dim=1)
                preds.extend(batch_preds.detach().cpu().numpy())

        return preds




