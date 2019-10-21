import os, shutil, time, pprint
import torch
import numpy as np

import itertools
import collections



from sklearn.metrics import f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ensure_path(path):
    '''if os.path.exists(path):
        if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
            shutil.rmtree(path)
            os.mkdir(path)
    else:'''
    os.mkdir(path)


def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 


def read_data(filename):
    content = []
    with open(filename, 'r') as data_src:
        tokens, tags = [], []
        for line in data_src:
            #print(line)
            if line.startswith("-DOCSTART-"): continue
            parts = line.strip().split()
            if len(parts) < 2:
                if len(tokens) > 0:
                    content.append((tokens, tags))
                    tokens, tags = [], []
                continue
            #print(parts)
            tokens.append(parts[0])
            if "B-" in parts[-1]:
                tags.append("I-"+parts[-1][2:])
            else:
                tags.append(parts[-1])
    return content



def simplify_token(token):
    chars = []
    for char in token:
        if char.isdigit():
            chars.append("0")
        else:
            chars.append(char)
    return ''.join(chars)



class Aggregate_F():
    def __init__(self):
        self.correct = {}
        self.total_preds = {}
        self.total_gold = {}

    def add(self, correct, total_preds, total_gold):
        Cdict = collections.defaultdict(int)
        for key, val in itertools.chain(self.correct.items(), correct.items()):
            Cdict[key] += val
        self.correct = dict(Cdict)

        Cdict = collections.defaultdict(int)
        for key, val in itertools.chain(self.total_preds.items(), total_preds.items()):
            Cdict[key] += val   
        self.total_preds = dict(Cdict)

        Cdict = collections.defaultdict(int)
        for key, val in itertools.chain(self.total_gold.items(), total_gold.items()):
            Cdict[key] += val
        self.total_gold = dict(Cdict)
        
    def item(self):
        return self.correct, self.total_preds, self.total_gold
        
    def f_score(self):
        precision = {k: self.correct[k] / max(float(self.total_preds[k]), 0.0001) for k in self.correct if k in self.total_preds}
        recall = {k: self.correct[k] / float(self.total_gold[k]) for k in self.correct if k in self.total_gold}
        
        f_score = {k: 2*(precision[k] * recall[k])/max(float(precision[k] + recall[k]), 0.0001) for k in precision if k in recall}
        return f_score


class Averager():    
    def __init__(self):
        self.n = 0
        self.v = 0
        
    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def pad_sentences(sentences, max_sent_len=100):
    max_len = 0
    lengths = []
    for _ in sentences: max_len = max(max_len, len(_))
    #max_len = min(max_len, max_sent_len)
    for i in range(len(sentences)):
        lengths.append(min(len(sentences[i]), max_len))
        sentences[i] = sentences[i] + [0]*(max_len - len(sentences[i]))
        sentences[i] = sentences[i][:max_len]
    return torch.LongTensor(sentences).to(device), torch.LongTensor(lengths).to(device)


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b) ** 2).sum(dim=2)
    return logits


def count_F(logits, labels, all_labels_set, counter=None):
    preds = torch.argmax(logits, dim=1)
    labels = labels.data.to('cpu')
    preds = preds.data.to('cpu')
    #return for each class, Correct_instance, Total Predicted, Total Gold
    correct_dict = dict.fromkeys(list(all_labels_set), 0)
    #print(correct_dict)
    total_predictions = dict.fromkeys(list(all_labels_set), 0)
    total_gold = dict.fromkeys(list(all_labels_set), 0)
    for pred, label in zip(preds, labels):
        #print(pred.item())
        #print(label.item())
        if pred == label:
            correct_dict[pred.item()] += 1
        total_predictions[pred.item()] += 1
        total_gold[label.item()] += 1
        
    return correct_dict, total_predictions, total_gold
    

def count_acc(logits, labels, counter=None):
    #compute F-score for each class
    preds = torch.argmax(logits, dim=1)
    #print(preds)
    #print("---")

    #indices = np.argwhere(np.array(counter) == 1)
    
    #labels = labels[indices]
    #preds = preds[indices]
    
    f1 = f1_score(labels.data.to('cpu'), preds.data.to('cpu'), average=None)
    print(f1)
    return torch.FloatTensor(f1[1:]).to(device).mean().item()
    #return (pred == label).type(torch.FloatTensor).to(device).mean().item()


def save_dev_span_output(dev_output_file, logits, labels, data_labels, data_sentence_query, data_token_query, query_shot_lens, id_to_token, id_to_tag):
    #For later aggregate them into the same sentence
    #dev_output_file = open(file_path, 'w')
    preds = torch.argmax(logits, dim=1)
    labels = labels.data.to('cpu')
    
    map_dict = dict()
    for l, dl in zip(labels, data_labels):
        map_dict[l.item()] = dl

    for (pred, label, data_label, data_sentence, data_token, len) in zip(preds, labels, data_labels, data_sentence_query, data_token_query, query_shot_lens):
        for i in range(len):
            #print(data_token)
            data_sentence = data_sentence.data.to('cpu')
            pred = pred.data.to('cpu')
            #data_label = data_label.data.to('cpu')
            #map between label and data_label
            #print(data_sentence)
            #print(pred)
            if i >= data_token[0] and i <= data_token[1]:
                #dev_output_file.write(str(id_to_token[data_sentence[i]])+"\t"+str(id_to_tag[map_dict[pred.item()]])+"\t"+str(id_to_tag[data_label])+"\n")
                dev_output_file.write(str(id_to_token[data_sentence[i]])+"\t"+str(id_to_tag[data_label])+"\t"+str(id_to_tag[map_dict[pred.item()]])+"\n")
            else:
                dev_output_file.write(str(id_to_token[data_sentence[i]])+"\n")
        dev_output_file.write("\n")
                                                                                                                                        
    

def save_dev_output(dev_output_file, logits, labels, data_labels, data_sentence_query, data_token_query, query_shot_lens, id_to_token, id_to_tag):
    #For later aggregate them into the same sentence
    #dev_output_file = open(file_path, 'w')
    preds = torch.argmax(logits, dim=1)
    labels = labels.data.to('cpu')            
    
    map_dict = dict()
    for l, dl in zip(labels, data_labels):
        map_dict[l.item()] = dl

    for (pred, label, data_label, data_sentence, data_token, len) in zip(preds, labels, data_labels, data_sentence_query, data_token_query, query_shot_lens):
        for i in range(len):
            #print(data_token)
            data_sentence = data_sentence.data.to('cpu')
            pred = pred.data.to('cpu')
            #data_label = data_label.data.to('cpu')
            #map between label and data_label
            #print(data_sentence)
            #print(pred)
            if i == data_token:
                #dev_output_file.write(str(id_to_token[data_sentence[i]])+"\t"+str(id_to_tag[map_dict[pred.item()]])+"\t"+str(id_to_tag[data_label])+"\n")
                dev_output_file.write(str(id_to_token[data_sentence[i]])+"\t"+str(id_to_tag[data_label])+"\t"+str(id_to_tag[map_dict[pred.item()]])+"\n")            
            else:
                dev_output_file.write(str(id_to_token[data_sentence[i]])+"\n")
        dev_output_file.write("\n")


def dot_metric(a, b):
    return torch.mm(a, b.t())

class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)


def l2_loss(pred, label):
    return ((pred - label)**2).sum() / len(pred) / 2



def get_segments(tags):
    """                                                                                                                        
    Segment-start symbols: `B-`, `U-`, and `S-`                                                                                  
    Segment-end symbols: `L-`, `E-`, `U-`, and `S-`                                                                               
    """
    starts, ends = {'B', 'U', 'S'}, {'L', 'E', 'U', 'S'}

    prev_tag = 'O'
    start_idx = -1
    segments, break_idices = [], set()
    for i, tag in enumerate(tags):
        if tag == 'O':
            prefix, tagc = 'O', 'O'
        elif '-' in tag:
            prefix, tagc = tag.split('-')
        else:
            continue
        if prefix in starts:
            break_idices.add(i)
        if prefix in ends:
            break_idices.add(i+1)
        if tagc != prev_tag:
            if prev_tag != 'O':
                segments.append((start_idx, i, prev_tag))
            start_idx = i
        prev_tag = tagc
                                                                                                        
    if prev_tag != 'O':
        segments.append((start_idx, len(tags), prev_tag))
    breaked_segments = []
    for s, e, t in segments:
        sidx = s
        for i in range(s+1, e):
            if i in break_idices:
                breaked_segments.append((sidx, i, t))
                sidx = i
        breaked_segments.append((sidx, e, t))
    return breaked_segments
                                                                                        
