# Reference: https://stackoverflow.com/questions/50815354/seq2seq-bidirectional-encoder-decoder-in-keras
# Attention layer:https://github.com/thushv89/attention_keras
# Inference model: https://www.analyticsvidhya.com/blog/2019/06/comprehensive-guide-text-summarization-using-deep-learning-python/

import matplotlib.pyplot as plt
import numpy as np
from attention import AttentionLayer
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Concatenate, TimeDistributed, Dense
from tensorflow.keras.models import Model
from keras import backend as K 
K.clear_session() 

class TextSummarizationModel:
    
    def __init__(self, x_tr, x_val, y_tr, y_val, voc, tokenizer):
        self.latent_dim = x_tr.shape[0]
        self.max_text_len = x_tr.shape[1]
        self.max_summary_len = y_tr.shape[1]

        self.model = None
        self.history = None

        self.voc = voc
        
        self.x_tr = x_tr
        self.x_val = x_val
        self.y_tr = y_tr
        self.y_val = y_val
        
        self.reverse_word_index = tokenizer.index_word
        self.word_index = tokenizer.word_index
        
        self.encoder_model = None
        self.decoder_model = None
    
    def create(self):
        
        # Encoder input
        encoder_inputs = Input(shape=(self.max_text_len,)) 
        
        # Embedding layer
        enc_emb = Embedding(self.voc, self.latent_dim, trainable=True)(encoder_inputs) 
        
        # Layers of bidirectional LSTM
        forward_layer = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        backward_layer = LSTM(self.latent_dim, return_sequences=True, return_state=True, go_backwards=True)
        
        # Bidirectional LSTM
        bidirectional_lstm = Bidirectional(forward_layer, backward_layer = backward_layer)
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = bidirectional_lstm(enc_emb)
        
        # Combining forward and backward LSTM states
        state_h = Concatenate()([forward_h, backward_h])
        state_c = Concatenate()([forward_c, backward_c])
        encoder_states = [state_h, state_c]
        
        # Decoder input
        decoder_inputs = Input(shape=(None,)) 
        
        # Embedding layer
        dec_emb_layer = Embedding(self.voc, self.latent_dim, trainable=True) 
        dec_emb = dec_emb_layer(decoder_inputs) 
        
        # Decoder LSTM
        decoder_lstm = LSTM(self.latent_dim*2, return_sequences=True, return_state=True)
        decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb, initial_state=[state_h, state_c]) 
        
        # Attention layer
        attention_layer = AttentionLayer(name = 'attention_layer') 
        attn_out, attn_states = attention_layer([encoder_outputs, decoder_outputs]) 
        
        decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])
        
        decoder_dense = TimeDistributed(Dense(self.voc, activation='softmax')) 
        decoder_outputs = decoder_dense(decoder_concat_input) 
        
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs) 
        print(self.model.summary())
        
        #--------------------------------------Inference------------------------------------------#
        
        self.encoder_model = Model(inputs = encoder_inputs, outputs = [encoder_outputs, state_h, state_c])

        # Decoder setup
        # Below tensors will hold the states of the previous time step
        decoder_state_input_h = Input(shape=(self.latent_dim*2,))
        decoder_state_input_c = Input(shape=(self.latent_dim*2,))
        decoder_hidden_state_input = Input(shape=(self.max_text_len, self.latent_dim*2))

        # Get the embeddings of the decoder sequence
        dec_emb2 = dec_emb_layer(decoder_inputs) 
        # To predict the next word in the sequence, set the initial states to the states from the previous time step
        decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

        #attention inference
        attn_out_inf, attn_states_inf = attention_layer([decoder_hidden_state_input, decoder_outputs2])
        decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

        # A dense softmax layer to generate prob dist. over the target vocabulary
        decoder_outputs2 = decoder_dense(decoder_inf_concat) 

        # Final decoder model
        self.decoder_model = Model(
            [decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
            [decoder_outputs2] + [state_h2, state_c2])
        
    def compile(self, optimizer, loss):
        
        self.model.compile(optimizer = optimizer, loss = loss)
        
    def fit(self, epochs = 30, validation_split = 0.2):
        
        self.history = self.model.fit([self.x_tr, self.y_tr[:,:-1]], self.y_tr.reshape(self.y_tr.shape[0], self.y_tr.shape[1], 1)[:,1:] , epochs = epochs, batch_size = int(0.1*self.latent_dim), validation_split = validation_split)
        
    def visualize_loss(self):
        
        plt.plot(self.history.history['loss'], label='Train')
        plt.plot(self.history.history['val_loss'], label='Validation loss')
        plt.legend()
        plt.show()
        
    def decode_sequence(self, input_seq):
        # Encode the input as state vectors.
        e_out, e_h, e_c = self.encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1,1))

        # Populate the first word of target sequence with the start word.
        target_seq[0, 0] = self.word_index['sostok']

        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:

            output_tokens, h, c = self.decoder_model.predict([target_seq] + [e_out, e_h, e_c])

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            
            try:
                sampled_token = self.reverse_word_index[sampled_token_index]
            except KeyError as e:
                sampled_token = 'unk'

            if(sampled_token!='eostok'):
                decoded_sentence += ' '+sampled_token

            # Exit condition: either hit max length or find stop word.
            if (sampled_token == 'eostok'  or len(decoded_sentence.split()) >= (self.max_summary_len-1)):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1,1))
            target_seq[0, 0] = sampled_token_index

            # Update internal states
            e_h, e_c = h, c

        return decoded_sentence
    
    def seq2summary(self, input_seq):
        newString=''
        for i in input_seq:
            if((i!=0 and i!=self.word_index['sostok']) and i!=self.word_index['eostok']):
                try:
                    newString=newString+self.reverse_word_index[i]+' '
                except KeyError:
                    newString=newString+'unk '
        return newString

    def seq2text(self, input_seq):
        newString=''
        for i in input_seq:
            if(i!=0):
                try:
                    newString=newString+self.reverse_word_index[i]+' '
                except KeyError:
                    newString=newString+'unk '
        return newString
    
    def get_summaries(self):
        
        original = open("Original.txt", "a+")
        created = open("Machine_created.txt", "a+")
        
        for i in range(0, self.x_val.shape[0]):
            print("Review:", self.seq2text(self.x_val[i]))
            print("Original summary:", self.seq2summary(self.y_val[i]))
            original.write(str(self.seq2summary(self.y_val[i])) + '\n')
            
            print("Predicted summary:", self.decode_sequence(self.x_val[i].reshape(1,self.max_text_len)))
            created.write(str(self.decode_sequence(self.x_val[i].reshape(1,self.max_text_len))) + '\n')
            
            print("\n")