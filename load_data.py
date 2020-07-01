import tensorflow_datasets as tfds
import pandas as pd
import urllib3
urllib3.disable_warnings()

class Dataset:
    def __init__(self):
        self.giga_train = None
        self.giga_df = None
    
    def load(self):
        self.giga_train = tfds.load('gigaword', split = 'train')
        
    def save(self, path, n_instances = 10000):
        
        dataset = self.giga_train.take(n_instances)
        
        giga_input = []
        giga_ref = []

        for instance in dataset:
            giga_input.append(str(instance['document'].numpy(), 'utf-8'))
            giga_ref.append(str(instance['summary'].numpy(), 'utf-8'))

        self.giga_df = pd.DataFrame()

        self.giga_df['Text'] = giga_input
        self.giga_df['Summary'] = giga_ref
        
        self.giga_df.to_csv(path, index = False)