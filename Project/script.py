import sys
language = sys.argv[1]    ## getting language from commandline
parser = sys.argv[2]      ## getting parser (DG/HPSG) from commandline

from library.preprocessing import *
from library.model import *
from library.parsing import *


path = 'data/'

## ------------ pre-processing ------------ 

filename = 'ner_data.csv'

transform = DataTransformation(path, filename, language)
transform.run()

## ------------ parsing ------------
if parser == 'DG':
    java_path = "C:/Program Files/Java/jre1.8.0_361/bin/java.exe"
    jar_path = 'C:/Users/natas/Downloads/IUB/Spring_2023/NER/stanford-parser/stanford-corenlp-4.2.2/stanford-corenlp-4.2.2.jar'
    models_jar_path = 'C:/Users/natas/Downloads/IUB/Spring_2023/NER/stanford-parser/stanford-corenlp-4.2.2-models-english.jar'
    special_tokens = ['[START]','[SEP]', '[TAG]', '[END]', '[UNK]', '[PAD]']
    max_len = 150

    dg = DependencyGrammar(java_path, jar_path, models_jar_path, path, language, special_tokens, max_len, parser)
    dg.run()
    
elif parser == 'HPSG':
    special_tokens = ['[START]','[SEP]', '[END]', '[UNK]', '[PAD]']
    max_len = 100
    model = 'HPSG'

    hpsg = HpsgGrammar(path, language, special_tokens, max_len, parser)
    hpsg.run()
    
## ------------ RNN model ------------
if parser == 'DG':
    max_len = 150
else:
    max_len = 100

epochs = 15
batch_size = 64

mdl = RNN_Model(path, language, max_len, parser, batch_size, epochs)
mdl.run()

## ------------ model evaluation ------------

pred_file_name = 'prediction.pkl'
gold_file_name = 'Y_test.pkl'

model_eval = EvaluateModel(path, model, pred_file_name, gold_file_name, language)
model_eval.run()