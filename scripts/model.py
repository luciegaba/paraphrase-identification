
import tensorflow as tf
from keras.layers import LSTM
from torch.optim import AdamW, SGD
import keras.backend as K
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

def manhattan_distance(A, B):
    """Calculates the Manhattan distance between two vectors A and B"""
    return K.sum(K.abs(A - B), axis=1, keepdims=True)
        
def euclidean_distance(A, B):
    """ Calculates the Euclidean distance between two vectors A and B"""
    return K.sqrt(K.sum(K.square(A - B), axis=1, keepdims=True))

def cosine_similarity_distance(A, B):
    """ Calculates the cosine similarity distance between two vectors A and B"""
    dot_product = K.sum(K.dot(A, K.transpose(B)), axis=1, keepdims=True)
    norm_A = K.sqrt(K.sum(K.square(A), axis=1, keepdims=True))
    norm_B = K.sqrt(K.sum(K.square(B), axis=1, keepdims=True))
    return dot_product / (norm_A * norm_B)

def contrastive_loss(y, d):
    """ Calculates the contrastive loss function using the target label y and the calculated distance d"""
    margin = 1
    return K.mean(y * K.square(d) + (1 - y) * K.square(K.maximum(margin - d, 0)))

def accuracy_score(y_true, y_pred):
    """
    Calculates the accuracy score given true labels and predicted labels.
    """
    y_pred = np.round(y_pred).astype(int)
    accuracy = np.mean(y_true == y_pred)
    return accuracy



class SiameseNet:
    """
    A class to represent a Siamese Neural Network for paraphrase identification.

    Attributes
    ----------
    input_dim : int
        Maximum length of input sentences
    weight_matrix : numpy array
        Pre-trained word embedding matrix
    bi_directional_architecture_option: boolean
        If True, use bi-directional layer
    hidden_layer_architecture : tensorflow.keras.layers.Layer
        Architecture of the hidden layer (default: LSTM)
    hidden_layer_neurons : int
        Number of neurons in the hidden layer (default: 128)
    activation_intermediate_option : bool
        Whether to add an intermediate activation layer (default: True)
    activation_intermediate_function : str
        Name of the activation function for the intermediate layer (default: "relu")
    distance_layer_function : function
        Function used to calculate the distance between encoded sentences (default: manhattan_distance)
    activation_function_output : str
        Name of the activation function for the output layer (default: "sigmoid")
    model : tensorflow.keras.models.Model
        Siamese Neural Network model

    Methods
    -------
    build_model()
        Builds the Siamese Neural Network model
    fit(train_sentences_1, train_sentences_2, train_labels, test_sentences_1, test_sentences_2, test_labels, batch_size=32, epochs=10, validation_split=0.2, verbose=1, cost_function="binary_crossentropy", metrics=["binary_crossentropy","accuracy"], optimizer=tf.keras.optimizers.Adam(learning_rate=0.003))
        Fits the Siamese Neural Network to the training data
    predict(sentences_1, sentences_2)
        Predicts the similarity score between pairs of input sentences
    """
    
    def __init__(self,max_len, weight_matrix,bi_directional_architecture_option=False, hidden_layer_architecture=LSTM,hidden_layer_neurons=128,distance_layer_function=manhattan_distance,activation_function_output = "sigmoid"):
        self.input_dim = max_len
        self. weight_matrix = weight_matrix
        self.input_dim_embedding = weight_matrix.shape[0]
        self.output_dim_embedding = weight_matrix.shape[1]
        self.bi_directional_architecture_option = bi_directional_architecture_option
        self.hidden_layer_architecture = hidden_layer_architecture
        self.hidden_layer_neurons = hidden_layer_neurons
        self.distance_layer_function = distance_layer_function
        self.activation_function_output = activation_function_output
        self.model = self.build_model()
        
    def build_model(self):
        """
        Builds and returns a Siamese network model.

        Returns
        -------
        model : keras.models.Model
            The Siamese network model.
        """
        #Define inputs (input 1 and 2)
        input1 = tf.keras.layers.Input(shape=(self.input_dim,))
        input2 = tf.keras.layers.Input(shape=(self.input_dim,))
        #Embedding layer using pre-trained weight matrix
        shared_embedding = tf.keras.layers.Embedding(input_dim=self.input_dim_embedding, output_dim=self.output_dim_embedding, weights=[self.weight_matrix])
        #Hidden layer 
        if self.bi_directional_architecture_option is True:
            shared_hidden_layer = tf.keras.layers.Bidirectional(self.hidden_layer_architecture(self.hidden_layer_neurons, return_sequences=True))
        else: 
            shared_hidden_layer = self.hidden_layer_architecture(self.hidden_layer_neurons)
        #Apply lstm on inputs
        encoded1 = shared_hidden_layer(shared_embedding(input1))
        encoded2 = shared_hidden_layer(shared_embedding(input2))
        #Concatenate distance between 1 and 2
        merged_vector = tf.keras.layers.Lambda(lambda x: self.distance_layer_function(x[0], x[1]))([encoded1, encoded2])
        #Output activation function
        output = tf.keras.layers.Dense(1, activation=self.activation_function_output)(merged_vector)
        siamese_net = tf.keras.models.Model(inputs=[input1, input2], outputs=output)
        return siamese_net
    
    def fit(self, train_sentences_1, train_sentences_2, train_labels,test_sentences_1,test_sentences_2,test_labels, batch_size=32, epochs=10, verbose=1,cost_function = "binary_crossentropy",metrics = ["binary_crossentropy","accuracy"]
,optimizer="adam"):
        """
        Trains the Siamese network model.

        Parameters
        ----------
        train_sentences_1 : numpy.ndarray or list
            The first set of training sentences.
        train_sentences_2 : numpy.ndarray or list
            The second set of training sentences.
        train_labels : numpy.ndarray 
            The labels for the training data.
        test_sentences_1 : numpy.ndarray or list
            The first set of test sentences.
        test_sentences_2 : numpy.ndarray or list
            The second set of test sentences.
        test_labels : numpy.ndarray
            The labels for the test data.
        batch_size : int, optional
            The batch size for training, by default 32.
        epochs : int, optional
            The number of epochs to train for, by default 10.
        validation_split : float, optional
            The percentage of the training data to use for validation, by default 0.2.
        verbose : int, optional
            The verbosity level for training output, by default 1.
        cost_function : str, optional
            The loss function to use during training, by default "binary_crossentropy".
        metrics : list of str, optional
            The metrics to track during training, by default ["binary_crossentropy", "accuracy"].
        optimizer : tensorflow.keras.optimizers.Optimizer, optional
            The optimizer to use during training, by default "adam".
        """
        
        self.model.compile(loss=cost_function, metrics=metrics,optimizer=optimizer)
        self.model.fit(
            [train_sentences_1, train_sentences_2], train_labels,
            batch_size=batch_size, epochs=epochs, verbose=verbose,
            validation_data=([test_sentences_1,test_sentences_2],test_labels)
        )
        

    def predict(self, sentences_1, sentences_2):
        """
        Makes predictions using the trained Siamese network model.

        Parameters
        ----------
        sentences_1 : numpy.ndarray or list
            The first set of sentences to compare.
        sentences_2 : numpy.ndarray or list
            The second set of sentences to compare.

        Returns
        -------
        numpy.ndarray
            The probability to be paraphrase 
        """
        return self.model.predict([sentences_1, sentences_2])






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

        Parameters:
        -----------
        batch : List[Tuple[torch.Tensor]]
            A batch of input sequences and labels.

        Returns:
        --------
        input_ids : torch.Tensor
            Tensor of shape (batch_size, max_seq_len) containing the input sequence tokens.
        attention_masks : torch.Tensor
            Tensor of shape (batch_size, max_seq_len) containing the attention masks.
        token_type_ids : torch.Tensor
            Tensor of shape (batch_size, max_seq_len) containing the token type ids.
        labels : torch.Tensor
            Tensor of shape (batch_size,) containing the labels.
        """
        input_ids, attention_masks, token_type_ids, labels = zip(*batch)
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
        labels = torch.tensor(labels, dtype=torch.long)
        return input_ids, attention_masks, token_type_ids, labels
    
    def fit(self, train_sentences_1, train_sentences_2, train_labels, optimizer_name='AdamW'):
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
        optimizer_name : str, optional
            The name of the optimizer to use for fine-tuning (default: 'AdamW').
            Supported optimizers: 'AdamW', 'SGD'.
        """
        #dataset = dataset.remove_columns(["sentences_1", "sentences_2"])

        train_dataset = self.create_dataset(train_sentences_1, train_sentences_2, train_labels)
        dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,collate_fn=self.custom_dataloader_padder, drop_last=True)

        optimizers = {
            'AdamW': AdamW(self.model.parameters(), lr=self.learning_rate),
            'SGD': SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        }

        optimizer = optimizers[optimizer_name]

        num_training_steps = len(dataloader) * self.num_epochs
        warmup_steps = int(0.1 * num_training_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)
        self.model.train()
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            epoch_preds, epoch_labels = [], []

            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}"):
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

            epoch_accuracy = accuracy_score(epoch_labels, epoch_preds)
            print(f'Epoch: {epoch+1}, Loss: {running_loss/len(dataloader):.4f}, Accuracy: {epoch_accuracy:.4f}')

    def predict(self, sentences_1, sentences_2):
        """
        Predicts the labels for the given sentence pairs.

        Parameters
        ----------
        sentences_1 : list of str
            The first set of sentences.
        sentences_2 : list of str
            The second set of sentences.

        Returns
        -------
        preds : list of int
            The predicted labels for each sentence pair.
        """

        input_ids, attention_masks, token_type_ids = self.encode_sentences(sentences_1, sentences_2)
        dataset = TensorDataset(input_ids, attention_masks, token_type_ids)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        self.model.eval()
        preds = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_masks, token_type_ids = batch
                outputs = self.model(input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)
                logits = outputs.logits
                batch_preds = torch.argmax(logits, dim=1)
                preds.extend(batch_preds.detach().cpu().numpy())

        return preds
