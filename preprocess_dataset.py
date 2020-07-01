import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

class PreprocessDataset:
    
    def __init__(self, path):
        self.path = path
        self.dataset = pd.read_csv(path, nrows=5000)
        self.x_tr = None
        self.x_val = None
        self.y_tr = None
        self.y_val = None
        
        self.voc = 0
        self.text_count = []
        self.summary_count = []
        
    def text_cleaner(self, text):
        text = text.lower()
        text = text.replace(".", "")
        text = text.replace("unk", "")
        text = text.replace(" 's", "")
        tokens = text.split()
        tokens = [t for t in tokens if t.isalnum()]
        
        return (" ".join(tokens)).strip()
    
    def clean_dataset(self):
        cleaned_text = []
        cleaned_summary = []
        
        for text in self.dataset['Text']:
            cleaned_text.append(self.text_cleaner(text)) 
            
        for text in self.dataset['Summary']:
            cleaned_summary.append(self.text_cleaner(text))
            
        self.dataset['Clean_text'] = cleaned_text
        self.dataset['Clean_summary'] = cleaned_summary
            
    def save(self, path):
        
        self.dataset.to_csv(path, index = False)
        
    def visualize_sentence_lengths(self):
        self.text_count = []
        self.summary_count = []
        
        for text, summary in zip(self.dataset['Clean_text'], self.dataset['Clean_summary']):
            self.text_count.append(len(text.split()))
            self.summary_count.append(len(summary.split()))
            
        lengths = pd.DataFrame({'Text':self.text_count, 'Summary':self.summary_count})
        lengths.hist(bins = 30)
        plt.show()
            
    def get_max_lengths(self):
        
        return max(self.text_count), max(self.summary_count)
    
    def add_start_end_token(self):
        
        self.dataset['Clean_summary'] = self.dataset['Clean_summary'].apply(lambda x: 'sostok '+ x + ' eostok')
        
    def split_dataset(self, test_split = 0.2):
        self.x_tr, self.x_val, self.y_tr, self.y_val = train_test_split(np.array(self.dataset['Clean_text']),np.array(self.dataset['Clean_summary']),test_size = test_split,shuffle=True) 
        
    def tokenize(self, max_text_length, max_summary_length):
        
        tokenizer = Tokenizer() 
        tokenizer.fit_on_texts(list(self.x_tr) + list(self.y_tr))

        text_tr_seq    =   tokenizer.texts_to_sequences(self.x_tr) 
        text_val_seq   =   tokenizer.texts_to_sequences(self.x_val)

        self.x_tr    =   pad_sequences(text_tr_seq, maxlen=max_text_length, padding='post')
        self.x_val   =   pad_sequences(text_val_seq, maxlen=max_text_length, padding='post')

        summary_tr_seq    =   tokenizer.texts_to_sequences(self.y_tr) 
        summary_val_seq   =   tokenizer.texts_to_sequences(self.y_val)

        self.y_tr    =   pad_sequences(summary_tr_seq, maxlen=max_summary_length, padding='post')
        self.y_val   =   pad_sequences(summary_val_seq, maxlen=max_summary_length, padding='post')

        self.voc   =  len(tokenizer.word_index) + 1
        
        return tokenizer
        
    def get_dataset(self):
        
        return self.x_tr, self.x_val, self.y_tr, self.y_val
    
    def get_voc_size(self):
        
        return self.voc