import os
import re
import pandas as pd
import pickle
from delphin.web import client
from nltk.parse import stanford
from nltk.parse.stanford import StanfordDependencyParser
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class DependencyGrammar():
    
    def __init__(self, java_path, jar_path, models_jar_path, path, language, special_tokens, max_len, model):
        super().__init__()
        self.in_path = path+language+'/'+'raw/'
        self.out_path = path+language+'/'+model+'/'
        self.java_path = java_path
        self.jar_path = jar_path
        self.models_jar_path = models_jar_path
        self.special_tokens = special_tokens
        self.max_len = max_len
    
        
    def run(self):
        
        ## ------------------------------------- Helper functions ------------------------------------------------------------##
        
        def read_pickle(file_name, path):
            with open(path+file_name, 'rb') as handle: 
                data = pickle.load(handle)
            return data
        
        def write_pickle(file_name, file_object, path):
            with open(path+file_name, 'wb') as file: 
                pickle.dump(file_object, file)        
        
        def get_parsing(parser, sentence):
            result = parser.raw_parse(sentence)
            dependency = result.__next__()
            try:
                dep_triplet = list(dependency.triples())
                return dep_triplet
            except:
                return ''
        
        def transform_seq(sentence, triplet_set):
            seq = '[START]'+ ' ' + sentence + ' ' + '[SEP]' + ' ' 
            if triplet_set != '':
                for triplet in triplet_set: 
                    head_word = triplet[0][0]
                    head_pos  = triplet[0][1]
                    relation = triplet[1]
                    dependent_word = triplet[2][0]
                    dependent_pos = triplet[2][1]
                    seq += triplet[0][0] + ' ' + triplet[0][1] + ' ' + triplet[1] + ' ' + triplet[2][0] + ' ' + triplet[2][1] + ' ' + '[TAG]' + ' '
                seq = seq[:-6] # to remove last '[TAG]'
            seq+= '[END]' 
            return seq
        
        def create_mapping(vocab, tags):
            word_idx = {w : i + 1 for i ,w in enumerate(vocab)} 
            tag_idx =  {t : i for i ,t in enumerate(tags)}  
            idx_word = {i : w for w, i in word_idx.items()}
            idx_tag  = {i : t for t, i in tag_idx.items()}
            return word_idx, tag_idx, idx_word, idx_tag 

        ## ------------------------------------- Main Execution ------------------------------------------------------------##
        
        ## ------- reading raw data ------- 
        corpus = read_pickle('corpus.pkl', self.in_path)
        tag_seq = read_pickle('tags.pkl', self.in_path)
        words = read_pickle('unique_words.pkl', self.in_path)
        tags = read_pickle('unique_tags.pkl', self.in_path)
        
        ## ------- reading data related to parser-------
        unique_tags = read_pickle('parser_unique_tags.pkl', self.out_path)
        unique_relations = read_pickle('parser_unique_relations.pkl', self.out_path)
        
        ## ------- initialize Parser -------
        os.environ['JAVAHOME'] = self.java_path
        parser = StanfordDependencyParser(path_to_jar = self.jar_path, path_to_models_jar = self.models_jar_path)
        
        ## ------- parsing and transforming -------
        transformed_data = []
        for sentence, tag in zip(corpus, tag_seq):
            dependency_triplet = get_parsing(parser, sentence)
            transformed_sentence = transform_seq(sentence, dependency_triplet)
            transformed_data.append((transformed_sentence, tag)) 
        
        ## ------- creating mapping -------
        vocab = list(set(list(words) + list(self.special_tokens) + list(unique_tags) + list(unique_relations)))
        word_idx, tag_idx, idx_word, idx_tag = create_mapping(vocab, tags)
        
        ## ------- creating dataframe for NER -------
        ner_df = pd.DataFrame(transformed_data, columns=['Transformed_Sentence', 'Tag_Sequence'])
        transformed_sentence = ner_df['Transformed_Sentence'].values
        tag_sequence = ner_df['Tag_Sequence'].values
        
        ## ------- encoding the data and padding it -------
        X = [[word_idx[word] if word in word_idx.keys() else word_idx['[UNK]'] for word in sentence.split()] for sentence in transformed_sentence]
        X=pad_sequences(maxlen=self.max_len,sequences=X,padding='post',value=word_idx['[PAD]'])
        Y=[[tag_idx[t] for t in tag.split()] for tag in tag_sequence]
        Y=pad_sequences(maxlen=self.max_len,sequences=Y,padding='post',value=tag_idx['O']) # padding with 'O'
        
        ## ------- encoding the data and padding it -------
        Y_ohe =[to_categorical(i,num_classes=len(tags)) for i in Y]
        
        ## ------- splitting the data -------
        X_train, X_test, Y_train, Y_test=train_test_split(X,Y_ohe,test_size=0.1,random_state=1)
        
        ## ------- writing files -------
        write_pickle('vocab.pkl', vocab, self.in_path) # writing vocab file
        write_pickle('vocab.pkl', vocab, self.out_path) # writing vocab file
        write_pickle('word_idx.pkl', word_idx, self.out_path) # writing word_idx file
        write_pickle('tag_idx.pkl', tag_idx, self.in_path) # writing tag_idx file
        write_pickle('idx_word.pkl', idx_word, self.out_path) # writing idx_word file
        write_pickle('idx_tag.pkl', idx_tag, self.in_path) # writing idx_tag file
        write_pickle('X_train.pkl', X_train, self.out_path) # writing input train file
        write_pickle('X_test.pkl', X_test, self.out_path) # writing input test file
        write_pickle('Y_train.pkl', Y_train, self.out_path) # writing labels train file
        write_pickle('Y_test.pkl', Y_test, self.out_path) # writing labels test file 
        
        
class HpsgGrammar():
    
    def __init__(self, path, language, special_tokens, max_len, model):
        super().__init__()
        self.in_path = path+language+'/'+'raw/'
        self.out_path = path+language+'/'+model+'/'
        self.special_tokens = special_tokens
        self.max_len = max_len
    
        
    def run(self):
        
        ## ------------------------------------- Helper functions ------------------------------------------------------------##
        
        def read_pickle(file_name, path):
            with open(path+file_name, 'rb') as handle: 
                data = pickle.load(handle)
            return data
        
        def write_pickle(file_name, file_object, path):
            with open(path+file_name, 'wb') as file: 
                pickle.dump(file_object, file)        
        
        def split_underscore(input_string):
            words = input_string.split()

            new_words = []
            for word in words:
                index = word.find('_')  # find index of first underscore in the word
                if index != -1:  # if underscore is found
                    new_word = word[:index] + ' ' + word[index+1:]  # replace underscore with space
                else:
                    new_word = word  # if no underscore is found, keep the word as it is
                new_words.append(new_word)

            new_string = ' '.join(new_words)
            return new_string
        
        def clean_result(res):
            res = res.replace(' _', ' ')
            res = res.replace('+', ' ')
            res = split_underscore(res)
            res = res.replace('/', ' ')
            cleaned_res = re.sub(r'\s{2,}', ' ', res)
            return cleaned_res
        
        def get_hpsg(sentence):
            try:
                response = client.parse(sentence, params={'mrs': 'json'})
                m = response.result(0).mrs()
                m_str = str(m)
                start_ind = m_str.find("(")
                end_ind = m_str.find(")")
                res = m_str[start_ind+1:end_ind]
                cleaned_res = clean_result(res)
            except:
                cleaned_res = ''
            return cleaned_res
        
        def transform_seq(corpus, vocab):
            new_seq = []
            for sentence in corpus:
                hpsg = get_hpsg(sentence)
                unique_words = hpsg.split(' ')
                vocab += unique_words
                new_seq.append('[START]'+ ' ' + sentence + ' ' + '[SEP]' + ' ' + hpsg + ' '+ '[END]')
            vocab = list(set(vocab + ['[START]', '[SEP]', '[END]', '[PAD]', '[UNK]']))
            return new_seq, vocab
        
        def create_mapping(vocab):
            word_idx = {w : i + 1 for i ,w in enumerate(vocab)} 
            idx_word = {i : w for w, i in word_idx.items()}
            return word_idx, idx_word

        ## ------------------------------------- Main Execution ------------------------------------------------------------##
        
        ## ------- reading raw data ------- 
        corpus = read_pickle('corpus.pkl', self.in_path)
        tag_seq = read_pickle('tags.pkl', self.in_path)
        vocab = read_pickle('vocab.pkl', self.in_path)

        
        ## ------- parsing and transforming -------
        transformed_sentence, vocab = transform_seq(corpus, vocab)
        
        ## ------- creating mapping -------
        word_idx, idx_word = create_mapping(vocab)
        tag_idx = read_pickle('tag_idx.pkl', self.in_path)
        idx_tag = read_pickle('idx_tag.pkl', self.in_path)
        tags = list(tag_idx.keys())
        
        
        ## ------- encoding the data and padding it -------
        X = [[word_idx[word] if word in word_idx.keys() else word_idx['[UNK]'] for word in sentence.split()] for sentence in transformed_sentence]
        X=pad_sequences(maxlen=self.max_len,sequences=X,padding='post',value=word_idx['[PAD]'])
        Y=[[tag_idx[t] for t in tag.split()] for tag in tag_seq]
        Y=pad_sequences(maxlen=self.max_len,sequences=Y,padding='post',value=tag_idx['O']) # padding with 'O'
        
        ## ------- encoding the data and padding it -------
        Y_ohe =[to_categorical(i,num_classes=len(tags)) for i in Y]
        
        ## ------- splitting the data -------
        X_train, X_test, Y_train, Y_test=train_test_split(X,Y_ohe,test_size=0.1,random_state=1)
        
        ## ------- writing files -------
        write_pickle('vocab.pkl', vocab, self.out_path) # writing vocab file
        write_pickle('word_idx.pkl', word_idx, self.out_path) # writing word_idx file
        write_pickle('idx_word.pkl', idx_word, self.out_path) # writing idx_word file
        write_pickle('X_train.pkl', X_train, self.out_path) # writing input train file
        write_pickle('X_test.pkl', X_test, self.out_path) # writing input test file
        write_pickle('Y_train.pkl', Y_train, self.out_path) # writing labels train file
        write_pickle('Y_test.pkl', Y_test, self.out_path) # writing labels test file     