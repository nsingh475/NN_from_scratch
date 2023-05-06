import re
import pickle
import numpy as np


class EvaluateModel():
    
    def __init__(self, path, model, pred_file_name, gold_file_name, language):
        super().__init__()
        self.raw = path+language+'/raw/'
        self.path = path+language+'/'+model+'/'
        self.pred_file_name = pred_file_name
        self.gold_file_name = gold_file_name
        self.model = model
        
    def run(self):
        
        ## ------------------------------------- Helper functions ------------------------------------------------------------##
        
        def read_pickle(file_name, path):
            with open(path+file_name, 'rb') as handle: 
                data = pickle.load(handle)
            return data
        
        def get_class_index(labels):
            argmax_labels = []
            for instance in labels:
                argmax_row = []
                for char in instance:
                    temp = np.argmax(char)
                    argmax_row.append(temp)
                argmax_labels.append(argmax_row)
            return argmax_labels
        
        def get_actual_labels(labels, unique_tags):
            actual_labels = []
            for instance in labels:
                actual_row = []
                for char in instance:
                    temp = unique_tags[char]
                    actual_row.append(temp)
                actual_labels.append(actual_row)
            return actual_labels
        
        def exact_match(pred_labels, gold_labels):
            match = 0
            num_instances = len(pred_labels) 
            for i in range(num_instances):
                if pred_labels[i] == gold_labels[i]:
                    match += 1
            return round(match*100/num_instances, 2)
        
        def transform_main_label(labels):
            transformed_labels = []
            num_instances = len(labels)
            for i in range(num_instances):
                temp = re.sub('B-[\w]+', 'B', labels[i])
                temp = re.sub('I-[\w]+', 'I', temp)
                transformed_labels.append(temp.split(' '))
            return transformed_labels
        
        def get_count_match(pred_instance, gold_instance, label):
            count, match = 0, 0
            instance_len = len(pred_instance)
            for i in range(instance_len):
                if gold_instance[i] == label:
                    count += 1
                    if pred_instance[i] == gold_instance[i]:
                        match += 1
            return count, match
        
        def correct_main_label(pred_labels, gold_labels):
            pred_main = transform_main_label(pred_labels)
            gold_main = transform_main_label(gold_labels)
            count_B, match_B = 0, 0
            count_I, match_I = 0, 0
            count_O, match_O = 0, 0
            num_instances = len(pred_labels)
            for i in range(num_instances):
                count, match = get_count_match(pred_main[i], gold_main[i], 'B')
                count_B += count
                match_B += match
                count, match = get_count_match(pred_main[i], gold_main[i], 'I')
                count_I += count
                match_I += match
                count, match = get_count_match(pred_main[i], gold_main[i], 'O')
                count_O += count
                match_O += match
            correct_B = round(match_B*100/count_B, 2) if count_B != 0 else 0
            correct_I = round(match_I*100/count_I, 2) if count_I != 0 else 0
            correct_O = round(match_O*100/count_O, 2) if count_O != 0 else 0
            return correct_B, correct_I, correct_O
        
        def transform_sub_label(labels):
            transformed_labels = []
            num_instances = len(labels)
            for i in range(num_instances):
                temp = re.sub('[\w]-tim', 'tim', labels[i])
                temp = re.sub('[\w]-gpe', 'gpe', temp)
                temp = re.sub('[\w]-geo', 'geo', temp)
                temp = re.sub('[\w]-nat', 'nat', temp)
                temp = re.sub('[\w]-eve', 'eve', temp)
                temp = re.sub('[\w]-art', 'art', temp)
                temp = re.sub('[\w]-org', 'org', temp)
                temp = re.sub('[\w]-per', 'per', temp)
                transformed_labels.append(temp.split(' '))
            return transformed_labels
        
        def correct_sub_label(pred_labels, gold_labels):
            pred_main = transform_sub_label(pred_labels)
            gold_main = transform_sub_label(gold_labels)
            count_tim, match_tim = 0, 0
            count_gpe, match_gpe = 0, 0
            count_geo, match_geo = 0, 0
            count_nat, match_nat = 0, 0
            count_eve, match_eve = 0, 0
            count_art, match_art = 0, 0
            count_org, match_org = 0, 0
            count_per, match_per = 0, 0
            num_instances = len(pred_labels)
            for i in range(num_instances):
                count, match = get_count_match(pred_main[i], gold_main[i], 'tim')
                count_tim += count
                match_tim += match
                count, match = get_count_match(pred_main[i], gold_main[i], 'gpe')
                count_gpe += count
                match_gpe += match
                count, match = get_count_match(pred_main[i], gold_main[i], 'geo')
                count_geo += count
                match_geo += match
                count, match = get_count_match(pred_main[i], gold_main[i], 'nat')
                count_nat += count
                match_nat += match
                count, match = get_count_match(pred_main[i], gold_main[i], 'eve')
                count_eve += count
                match_eve += match
                count, match = get_count_match(pred_main[i], gold_main[i], 'art')
                count_art += count
                match_art += match
                count, match = get_count_match(pred_main[i], gold_main[i], 'org')
                count_org += count
                match_org += match
                count, match = get_count_match(pred_main[i], gold_main[i], 'per')
                count_per += count
                match_per += match
            correct_tim = round(match_tim*100/count_tim, 2) if count_tim != 0 else 0
            correct_gpe = round(match_gpe*100/ count_gpe, 2) if count_gpe != 0 else 0
            correct_geo = round(match_geo*100/ count_geo, 2) if count_geo != 0 else 0
            correct_nat = round(match_nat*100/ count_nat, 2) if count_nat != 0 else 0
            correct_eve = round(match_eve*100/ count_eve, 2) if count_eve != 0 else 0
            correct_art = round(match_art*100/ count_art, 2) if count_art != 0 else 0
            correct_org = round(match_org*100/ count_org, 2) if count_org != 0 else 0
            correct_per = round(match_per*100/ count_per, 2) if count_per != 0 else 0
            return correct_tim, correct_gpe, correct_geo, correct_nat, correct_eve, correct_art, correct_org, correct_per
        
        
        ## ------------------------------------- Main Execution ------------------------------------------------------------##
        
        ## reading the predictions and gold_labels
        pred_labels = read_pickle(self.pred_file_name, self.path)
        gold_labels = read_pickle(self.gold_file_name, self.path)
        
        ## getting the most probable class from softmax output
        pred_labels = get_class_index(pred_labels)
        gold_labels = get_class_index(gold_labels)
        
        ## extracting the actual labels fro the class
        idx_tag = read_pickle('idx_tag.pkl', self.raw)
        pred_labels = get_actual_labels(pred_labels, idx_tag)
        gold_labels = get_actual_labels(gold_labels, idx_tag)
        
        pred_labels_str = [ ' '.join(sub_pred) for sub_pred in pred_labels]
        gold_labels_str = [ ' '.join(sub_gold) for sub_gold in gold_labels]
        
        f = open(f"{self.path}EvaluationOutput.txt", "a")
        print(f"------------------ {self.model} model Evaluation Report ------------------", file=f)
        res = exact_match(pred_labels_str, gold_labels_str)
        print(f"Exact Match percentage of Output labels: {res}", file=f)
        res = correct_main_label(pred_labels_str, gold_labels_str)
        print(f"Beginning label (B) match percent: {res[0]}", file=f)
        print(f"Inside label (I) match percent: {res[1]}", file=f)
        print(f"Outside label (O) match percent: {res[2]}", file=f)
        res = correct_sub_label(pred_labels_str, gold_labels_str)
        print(f"Time sub-label (tim) match percent: {res[0]}", file=f)
        print(f"Geopolitical Entity sub-label (gpe) match percent: {res[1]}", file=f)
        print(f"Geography sub-label (geo) match percent: {res[2]}", file=f)
        print(f"Economic Value Equity sub-label (eve) match percent: {res[4]}", file=f)
        print(f"Organization sub-label (org) match percent: {res[6]}", file=f)
        print(f"Person sub-label (per) match percent: {res[7]}", file=f)
        f.close()
        
