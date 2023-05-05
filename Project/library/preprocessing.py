import pandas as pd
import pickle


class DataTransformation():
    
    def __init__(self, path, filename, language):
        super().__init__()
        self.path = path+language+'/'+'raw/'
        self.filename = filename
        
    def run(self):
        
        ## ------------------------------------- Helper functions ------------------------------------------------------------##
        
        def collecting_sequence(sent_num, words, ner_tags):
            corpus = []
            tag_seq = []
            sent_collecting = 'Sentence: 1'
            current_sent = ''
            tag_sent = ''

            for num, word, tag in zip(sent_num, words, ner_tags):
                if num == sent_collecting:
                    # append to current sentence and tag sequence
                    current_sent += word + ' '
                    tag_sent += tag + ' '
                else:
                    # append old sentence and tag sequence
                    corpus.append(current_sent.strip())
                    tag_seq.append(tag_sent.strip())
                    # reset sentence and tag sequence
                    current_sent = word + ' '
                    tag_sent = tag + ' '
                    sent_collecting = num
            return corpus, tag_seq
            
        def write_pickle(file_name, file_object, path):
            with open(path+file_name, 'wb') as file: 
                pickle.dump(file_object, file)
        
        
        ## ------------------------------------- Main Execution ------------------------------------------------------------##
        
        ## ------- reading raw data ------- 
        ner_data = pd.read_csv(self.path+self.filename, encoding='unicode_escape')
        ner_data.drop('POS',axis='columns',inplace=True)
        ner_data=ner_data.fillna(method='ffill')
        
        ## ------- getting unique words and tags -------
        words=list(set(ner_data['Word'].values))
        tags = list(set(ner_data["Tag"].values))
        
        ## ------- collecting all column values in a list-------
        sent_num = ner_data['Sentence #'].values
        words = ner_data['Word'].values
        ner_tags = ner_data['Tag'].values
        
        ## ------- collecting sequence of one sentence in one row -------
        corpus, tag_seq = collecting_sequence(sent_num, words, ner_tags)
        
        ## ------- writing files -------
        write_pickle('corpus.pkl', corpus, self.path) # writing input file
        write_pickle('tags.pkl', tag_seq, self.path) # writing tags file
        write_pickle('unique_words.pkl', words, self.path) # writing unique words file
        write_pickle('unique_tags.pkl', tags, self.path) # writing unique tags file
        
    
         