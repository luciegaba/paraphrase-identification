
import tensorflow as tf
from keras.layers import LSTM
import keras.backend as K
import numpy as np


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
    distance_layer_function : function
        Function used to calculate the distance between encoded sentences (default: manhattan_distance)
    activation_function_output : str
        Name of the activation function for the output layer (default: "sigmoid")
    dropout_rate : int
        Rate for dropout layer
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
    
    def __init__(self,max_len, weight_matrix,bi_directional_architecture_option=False, hidden_layer_architecture=LSTM,hidden_layer_neurons=128,distance_layer_function=manhattan_distance,activation_function_output = "sigmoid",dropout_rate=0.1):
        self.input_dim = max_len
        self. weight_matrix = weight_matrix
        self.input_dim_embedding = weight_matrix.shape[0]
        self.output_dim_embedding = weight_matrix.shape[1]
        self.bi_directional_architecture_option = bi_directional_architecture_option
        self.hidden_layer_architecture = hidden_layer_architecture
        self.hidden_layer_neurons = hidden_layer_neurons
        self.distance_layer_function = distance_layer_function
        self.activation_function_output = activation_function_output
        self.dropout_rate = dropout_rate
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
            shared_hidden_layer = tf.keras.layers.Bidirectional(self.hidden_layer_architecture(self.hidden_layer_neurons, return_sequences=False))
        else: 
            shared_hidden_layer = self.hidden_layer_architecture(self.hidden_layer_neurons)
        dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)
        #Apply lstm on inputs
        encoded1 = shared_hidden_layer(shared_embedding(input1))
        encoded2 = shared_hidden_layer(shared_embedding(input2))
        #Concatenate distance between 1 and 2
        dropout_layer_s1 = dropout_layer(encoded1)
        dropout_layer_s2 = dropout_layer(encoded2)
        merged_vector = tf.keras.layers.Lambda(lambda x: self.distance_layer_function(x[0], x[1]))([dropout_layer_s1, dropout_layer_s2])
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
        history = self.model.fit(
            [train_sentences_1, train_sentences_2], train_labels,
            batch_size=batch_size, epochs=epochs, verbose=verbose,
            validation_data=([test_sentences_1,test_sentences_2],test_labels),
        )
        return history
        

    def predict(self, sentences_1, sentences_2, labels=None, threshold=0.5):
        """
        Makes predictions using the trained Siamese network model.

        Parameters
        ----------
        sentences_1 : numpy.ndarray or list
            The first set of sentences to compare.
        sentences_2 : numpy.ndarray or list
            The second set of sentences to compare.
        labels : numpy.ndarray or list, optional
            The labels corresponding to the sentences, by default None.
        threshold : float, optional
            The threshold for determining paraphrase or not, by default 0.5

        Returns
        -------
        numpy.ndarray
            The predicted labels (0 for non-paraphrase, 1 for paraphrase)
        """
        probabilities = self.model.predict([sentences_1, sentences_2])
        pred_labels = np.where(probabilities>threshold,1,0)

        return pred_labels









