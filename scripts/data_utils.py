import re
import pandas as pd
import nltk
import contractions
from nltk.stem import WordNetLemmatizer
from datasets import load_dataset

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models.keyedvectors import load_word2vec_format
import fasttext.util
import numpy as np


nltk.download('wordnet')
nltk.download('punkt')


class HuggingFaceExtracting():
    """
        A class to represent a dataset extractor for Hugging Face datasets.

        Attributes
        ----------
        bib : str
            bibliography string of the dataset
        dataset : str
            dataset string name
        regex_method_inputs : function
            regex function for inputs extraction
        export_option : bool
            flag for exporting extracted data to csv
        export_folder : str
            path to export the extracted data

        Methods
        -------
        huggingface_inputs_regex(input_pairs):
            Extract inputs from input pairs using regex.
            
        huggingface_labels_encoding(label):
            Encode labels into numerical values.
            
        load_hugging_face_dataset():
            Load Hugging Face dataset using load_dataset function.
            
        extract(split_sample, inputs_columns_name="inputs_pretokenized", labels_columns_name="targets_pretokenized"):
            Extract input pairs and labels from the dataset as "inputs" and "labels" columns
            
        """

    def __init__(self,bib="bigscience/P3",dataset="glue_qqp_same_thing",regex_method_inputs=None,export_option=False,export_folder=""):

        self.bib = bib
        self.dataset = dataset
        self.regex_method_inputs = regex_method_inputs
        self.export_option =export_option
        self.export_folder = export_folder
        
        
    def huggingface_inputs_regex(self,input_pairs:str):
        list_from_regex = list(re.search('(?<=")(.*)(?:" and ")(.*)(?=." asking the same thing)',input_pairs).groups())
        return list_from_regex
    
    def huggingface_labels_encoding(self,label:str):
 
        if label == " yes":
            label = 1
        elif label == " no":
            label = 0
        return label

    def load_huggingface_dataset(self):
        self.dataset = load_dataset(self.bib,self.dataset)
        
    def extract(self,split_sample:str,inputs_columns_name="inputs_pretokenized",labels_columns_name="targets_pretokenized"):

        raw_data = self.dataset[split_sample]   
        raw_data = pd.DataFrame(raw_data)     
        if self.bib ==  "P3":
            inputs_columns_name = "inputs_pretokenized"
            labels_columns_name = "targets_pretokenized"
        else:
            assert "Please give input and label columns in extract params"
        if self.regex_method_inputs == None:
            self.regex_method_inputs = self.huggingface_inputs_regex
            self.regex_method_inputs = self.huggingface_labels_encoding
        raw_data["inputs"] = raw_data[inputs_columns_name].apply(self.huggingface_inputs_regex)
        raw_data["labels"] = raw_data[labels_columns_name].map(self.huggingface_labels_encoding)
        if self.export_option is True:
            raw_data.to_csv(f"{self.bib}_{self.dataset}_{self.split}_data.csv")
        else:
            return raw_data[["inputs","labels"]]
            
            
                
class ETLPipeline():
    """
    A class representing an ETL pipeline for text datasets.

    Attributes
    ----------
    join_option : bool
        If True, the extracted input sentences will be joined into a single string.
    lemmatize_option : bool
        If True, the extracted input sentences will be lemmatized.
    output_type : str
        The type of output to produce: "list" (default) for Python lists, or "array" for NumPy arrays.

    Methods
    -------
    standardize_text_inputs(input_data)
        Processes raw input sentences by converting to lowercase, expanding contractions, tokenizing, and lemmatizing (if enabled).
        
    transform(data, label_column_name="labels", input_column_name="inputs")
        Transforms a dataset of sentence pairs and labels into inputs suitable for a text classification model.

    """  
    
    def __init__(self,join_option=True,lemmatize_option=True,output_type="list"):
        self.join_option = join_option
        self.lemmatize_option = lemmatize_option
        self.output_type = output_type
    

    def standardize_text_inputs(self, input_data):
        self.lemmatizer = WordNetLemmatizer()
        
        def process_sentence(sentence):
            sentence = contractions.fix(sentence)
            sentence = sentence.replace('"', "").replace('?', "")
            tokens = nltk.word_tokenize(sentence.lower())
            if self.lemmatize_option is True:
                return [self.lemmatizer.lemmatize(word) for word in tokens if word.isalpha()]
            else:
                return [word for word in tokens if word.isalpha()]

        all_tokens = [process_sentence(sentence) for sentence in input_data]

        if self.join_option:
            all_tokens = [" ".join(tokens) for tokens in all_tokens]
            
        return all_tokens
        
    
    
    def transform(self,data,label_column_name = "labels", input_column_name = "inputs"):

        self.labels = data[label_column_name].values.tolist()
        self.pairs_data = data[input_column_name].values.tolist()
        self.sentences_1,self.sentences_2 = zip(*self.pairs_data)
        self.sentences_1 = self.standardize_text_inputs(self.sentences_1)
        self.sentences_2 = self.standardize_text_inputs(self.sentences_2)
        if self.output_type == "list":
            inputs = {"sentences_1": self.sentences_1,
                "sentences_2": self.sentences_2,
                "label": self.labels,
                
            }
        elif self.output_type == "array":
           inputs = {"sentences_1": np.array(self.sentences_1),
                "sentences_2": np.array(self.sentences_2),
                "label": np.array(self.labels),
                
            } 
            
        return inputs
    
    
class WordEmbedding():
    """
    A class representing a word embedding model.

    Attributes
    ----------
    max_len : int
        The maximum length of input sequences to use for padding.
    model_word_embedding : str
        The type of word embedding model to use ("Word2Vec" or "FastText").
    model_path : str
        The path to the pre-trained word embedding model.
    pad_option : bool
        If True, input sequences will be padded to the specified maximum length.

    Methods
    -------
    tokenizer(train_inputs)
        Fits a tokenizer on the training inputs.
        
    tokenize_and_pad(sentences, **kwargs)
        Tokenizes and pads input sequences to the specified maximum length.
        
    load_pretrained_model()
        Loads a pre-trained word embedding model from a file.
        
    build_weight_matrix()
        Builds a weight matrix from the pre-trained word embedding model and the tokenizer.
        
    """


    def __init__(self,max_len=15,model_word_embedding=None,model_path="models/cc.en.100.bin",pad_option = True):
        self.max_len = max_len
        self.model_word_embedding = model_word_embedding
        self.model_path = model_path
        self.pad_option = pad_option

    def tokenizer(self, train_inputs):
            self.tokenizer = Tokenizer()
            self.tokenizer.fit_on_texts(train_inputs)
        
    def tokenize_and_pad(self,sentences,**kwargs):
        sentences = self.tokenizer.texts_to_sequences(sentences)
        if self.pad_option is True:  
            sentences = pad_sequences(sentences, maxlen=self.max_len,**kwargs)
        return sentences
    
    def load_pretrained_model(self):
        if self.model_word_embedding == "Word2Vec":
            self.pretrained_model = load_word2vec_format.load(self.model_path)
            self.pretrained_model = self.pretrained_model.wv

        elif self.model_word_embedding == "FastText":
            self.pretrained_model =  fasttext.load_model(self.model_path)
            self.embedding_size = self.pretrained_model.get_dimension()

            
    def build_weight_matrix(self):
        vocab_size = len(self.tokenizer.word_index) + 1
        weight_matrix = np.zeros((vocab_size, self.embedding_size))
        for word, i in self.tokenizer.word_index.items():
            try:
                embedding_vector = self.pretrained_model[word]
                weight_matrix[i] = embedding_vector
            except KeyError:
                weight_matrix[i] = np.random.uniform(-5, 5, self.embedding_size)
        return weight_matrix
    

