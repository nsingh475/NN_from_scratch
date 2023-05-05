import numpy as np
import pickle
from numpy.random import seed
from itertools import chain
from tensorflow.keras import Model,Input
from tensorflow.keras.layers import LSTM,Embedding,Dense
from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D,Bidirectional

class RNN_Model():
    
    def __init__(self, path, language, max_len, model_name, batch_size, epochs):
        super().__init__()
        self.in_path = path+language+'/'+'raw/'
        self.out_path = path+language+'/'+model_name+'/'
        self.max_len = max_len
        self.model_name = model_name
        self.batch_size = batch_size
        self.epochs = epochs
    
        
    def run(self):
        
        ## ------------------------------------- Helper functions ------------------------------------------------------------##
        
        def read_pickle(file_name, path):
            with open(path+file_name, 'rb') as handle: 
                data = pickle.load(handle)
            return data
        
        def write_pickle(file_name, file_object, path):
            with open(path+file_name, 'wb') as file: 
                pickle.dump(file_object, file)
                
        def build(num_words, max_len, num_tags):
            input_word=Input(shape=(max_len,))
            model=Embedding(input_dim=num_words,output_dim=max_len,input_length=max_len)(input_word)
            model=SpatialDropout1D(0.1)(model)
            model=Bidirectional(LSTM(units=100,return_sequences=True,recurrent_dropout=0.1))(model)
            model = Bidirectional(LSTM(50, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(model)
            out=TimeDistributed(Dense(num_tags,activation='softmax'))(model)
            return input_word, model, out
        

        ## ------------------------------------- Main Execution ------------------------------------------------------------##
        
        ## ------- reading raw data -------
        vocab = read_pickle('vocab.pkl', self.out_path)
        tags = read_pickle('unique_tags.pkl', self.in_path)
        num_words = len(vocab)
        num_tags = len(tags)
        
        X_train = read_pickle('X_train.pkl', self.out_path) 
        X_test = read_pickle('X_test.pkl',self.out_path) 
        Y_train = read_pickle('Y_train.pkl', self.out_path) 
        Y_test = read_pickle('Y_test.pkl', self.out_path) 
        
        ## ------- Building the model ------- 
        input_word, model, out = build(num_words, self.max_len, num_tags)
        model = Model(input_word,out)
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy', 'Precision', 'Recall'])
        model.fit(X_train,np.array(Y_train),batch_size=self.batch_size,verbose=1,epochs=self.epochs,validation_split=0.2)
        
        ## ------- model evaluation ------- 
        pred = model.predict(X_test)
        result = model.evaluate(X_test,np.array(Y_test)) 
        
        ## ------- writing model and results -------
        model.save(self.model_name, self.out_path)
        write_pickle('prediction.pkl', pred, self.out_path)
        write_pickle('actual_labels.pkl', np.array(Y_test), self.out_path)
        write_pickle('results.pkl', result, self.out_path) # writing evaluation results
    