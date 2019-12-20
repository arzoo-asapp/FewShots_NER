import os, shutil, time, pprint
import torch
import numpy as np

import itertools
import collections



from sklearn.metrics import f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ensure_path(path):
    if os.path.exists(path):
        #if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)


def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 


def read_bert_data(filename, bert_filename):
    content = []
    data_src = open(filename, 'r')
    bert_src = open(bert_filename, 'r')

    tokens, tags, bert_rep = [], [], []

    counter  = 0
    for line1, line2 in zip(data_src, bert_src):
        if line1.startswith("-DOCSTART-"): continue
        parts = line1.strip().split()

        if len(parts) < 2:
            if len(tokens) > 0:
                #print(len(tokens))
                #print(len(tags))
                #print(len(bert_rep))
                #print(tokens)
                #print(tags)
                #print(bert_rep)
                content.append((tokens, tags, bert_rep))
                counter += 1
                #if counter > 5000:
                #    break
                tokens, tags, bert_rep = [], [], []
            continue
        tokens.append(parts[0])
        if "B-" in parts[-1]:
            #or "I-" in parts[-1]:
            tags.append("I-"+parts[-1][2:])
            #tags.append(parts[-1][2:])
        else:
            #tags.append(parts[-1])
            if parts[-1] == "":
                print(tokens)
                print(tags)
                print(bert_rep)
                print("Upper mismatch")
                print("-----")
                exit()
            tags.append(parts[-1])
            #print(tags)
        bert_word = line2.split("\t")[0]
        if bert_word.lower() != parts[0].lower():
            print(parts[0]+"\t"+bert_word)
            print("Exit")
            exit()
        else:
            #print(line2.split("\t")[1:])
            bert_emb_list = list(line2.strip().split("\t")[1:])
            bert_emb_list = [float(emb) for emb in bert_emb_list]
            bert_rep.append(bert_emb_list)
            #print(len(list(line2.strip().split("\t")[1:])))
    print(counter)
    #exit()
    return content


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
                #or "I-" in parts[-1]:
                tags.append("I-"+parts[-1][2:])
                #tags.append(parts[-1][2:])
            else:
                #tags.append(parts[-1])
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
        recall = {k: self.correct[k] / max(float(self.total_gold[k]), 0.0001) for k in self.correct if k in self.total_gold}
        
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
        #print(sentences[i])
        sentences[i] = sentences[i] + [0]*(max_len - len(sentences[i]))
        sentences[i] = sentences[i][:max_len]
    return torch.LongTensor(sentences).to(device), torch.LongTensor(lengths).to(device)


def pad_query_sentences(sentences, labels, bert_emb, max_sent_len=100, PAD_CLS=-1):
    max_len = 0
    lengths = []
    for _ in sentences: max_len = max(max_len, len(_))
    #max_len = min(max_len, max_sent_len)
    #print(len(sentences))
    #print(max_len)
    for i in range(len(sentences)):
        lengths.append(min(len(sentences[i]), max_len))
        #print(sentences[i])
        sentences[i] = sentences[i] + [0]*(max_len - len(sentences[i]))
        labels[i] = labels[i] + [PAD_CLS]*(max_len - len(labels[i]))
        #print(len([0.]*768))
        #print(max_len)
        #print(len(labels[i]))
        #print([[0.]*768]*(max_len - len(labels[i])))
        #print("---")
        bert_emb[i] = bert_emb[i] + [[0.]*768]*(max_len - len(bert_emb[i]))
        sentences[i] = sentences[i][:max_len]
        labels[i] = labels[i][:max_len]
        bert_emb[i] = bert_emb[i][:max_len]
    #print(bert_emb)
    new_tensor = (torch.FloatTensor(bert_emb).to(device))
    #print(new_tensor.size())
    return torch.LongTensor(sentences).to(device), torch.LongTensor(labels).to(device), torch.FloatTensor(bert_emb).to(device), torch.LongTensor(lengths).to(device)
                                            


def euclidean_metric(a, b):
    '''print("Before A")
    #print(a)
    print(a.size())
    print("Before B")
    #print(b)
    print(b.size())'''
    n = a.shape[0]
    p = a.shape[1]
    m = b.shape[0]
    a = a.unsqueeze(2).expand(n, p, m, -1)
    b = b.unsqueeze(0).unsqueeze(1).expand(n, p, m, -1)
    logits = -((a - b) ** 2).sum(dim=3)
    #print("Logits")
    #print(logits)
    return logits



def count_F(logits, labels, all_labels_set, counter=None, train=True, PAD_CLS=-1, id_to_tag=None):
    confidence_labels = []
    
    preds = torch.argmax(logits, dim=2)
    pred_scores = torch.max(logits, dim=2)[0]
    labels = labels.data.to('cpu')
    preds = preds.data.to('cpu')
    #print(preds)
    #print(pred_scores)
    #exit()
    print(set(all_labels_set))
    #return for each class, Correct_instance, Total Predicted, Total Gold
    correct_dict_p = dict.fromkeys([id_to_tag[l] for l in list(all_labels_set)], 0)
    correct_dict_g = dict.fromkeys([id_to_tag[l] for l in list(all_labels_set)], 0)
        
    #print(correct_dict)
    total_predictions = dict.fromkeys([id_to_tag[l] for l in list(all_labels_set)], 0)
    total_gold = dict.fromkeys([id_to_tag[l] for l in list(all_labels_set)], 0)

    id_to_tag[PAD_CLS] = 'O'
    
    for pred, label, pred_score in zip(preds, labels, pred_scores):
        confidence_labels.append((pred_score, label, pred))
        #print(pred)
        new_pred = []
        new_label = []
        for p, l in zip(pred, label):
            if l.item() == PAD_CLS:
                continue
            if id_to_tag[p.item()] != 'O':
                new_pred.append("I-"+id_to_tag[p.item()])
            else:
                new_pred.append(id_to_tag[p.item()])

            if id_to_tag[l.item()] != 'O':
                new_label.append("I-"+id_to_tag[l.item()])
            else:
                new_label.append(id_to_tag[l.item()])
                
                
        #new_pred = [id_to_tag[p.item()] for p in pred]
        #print(new_pred)
        #new_label = [id_to_tag[l.item()] for l in label]
        #print(label)
        #print(new_label)
        #exit()
        pred_segments = get_segments(new_pred)
        gold_segments = get_segments(new_label)

        for psegment in pred_segments:
            (s1, e1, plabel) = psegment
            total_predictions[plabel] += 1
            for gsegment in gold_segments:
                (s2, e2, glabel) = gsegment
                if (s1 == s2 and e1 == e2 and plabel == glabel):
                    correct_dict_p[plabel] += 1
                    break

        for gsegment in gold_segments:
            (s1, e1, glabel) = gsegment
            total_gold[glabel] += 1
            for psegment in pred_segments:
                (s2, e2, plabel) = psegment
                if (s1 == s2 and e1 == e2 and plabel == glabel):
                    correct_dict_g[glabel] += 1
                    break

    return correct_dict_p, total_predictions, total_gold, confidence_labels
    
#exit()
        

def count_F_old(logits, labels, all_labels_set, counter=None, train=True, PAD_CLS=-1):
    confidence_labels = []
    preds = torch.argmax(logits, dim=2)
    pred_scores = torch.max(logits, dim=2)[0]
    labels = labels.data.to('cpu')
    preds = preds.data.to('cpu')
    #print(preds)
    #print(pred_scores)
    #exit()
    print(set(all_labels_set))
    #return for each class, Correct_instance, Total Predicted, Total Gold
    correct_dict = dict.fromkeys(list(all_labels_set), 0)
    #print(correct_dict)
    total_predictions = dict.fromkeys(list(all_labels_set), 0)
    total_gold = dict.fromkeys(list(all_labels_set), 0)
    for pred, label, pred_score in zip(preds, labels, pred_scores):
        confidence_labels.append((pred_score, label, pred))
        #print(pred)
        #print(label)
        
        #print(pred)
        #print(pred_score)
        #if pred > 0 and train:
        ###print(str(pred)+"\t"+str(pred_score)+"\t"+str(pred == label))
        #if pred > 0 and ((not train) and pred_score < 0.9):
        #    pred = torch.tensor(0)
        #    #print("Changed")

        for p, l in zip(pred, label):
            if l.item() == PAD_CLS:
                continue
            if p == l:
                correct_dict[p.item()] += 1
            total_predictions[p.item()] += 1
            total_gold[l.item()] += 1
                    
        
        '''if pred == label:
            #print(str(pred)+"\t"+str(pred_score))
            correct_dict[pred.item()] += 1
        total_predictions[pred.item()] += 1
        total_gold[label.item()] += 1'''

    return correct_dict, total_predictions, total_gold, confidence_labels
    

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


def save_dev_span_output(dev_output_file, logits, labels, new_data_labels, data_sentence_query, data_token_query, query_shot_lens, id_to_token, id_to_tag, test_cls = None):
    #For later aggregate them into the same sentence
    #dev_output_file = open(file_path, 'w')
    preds = torch.argmax(logits, dim=2)
    #print(preds.size())
    pred_scores = torch.max(logits, dim=2)[0]
    #print(pred_scores.size())
    #labels = labels.data.to('cpu')
    
    map_dict = dict()
    for l, dl in new_data_labels.items():
        map_dict[dl] = l

    #print(map_dict)
    #print("Batch_size")
    #print(len(preds))

    for (pred, pred_score, label, data_sentence, data_token, lens) in zip(preds, pred_scores, labels, data_sentence_query, data_token_query, query_shot_lens):
        #print(lens)
        for i in range(lens):
            #print(str(i)+"---->"+str(lens))
            #print(data_token)
            data_sentence = data_sentence.data.to('cpu')
            pred = pred.data.to('cpu')
            #data_label = data_label.data.to('cpu')
            #map between label and data_label
            #print(data_sentence)
            #print(pred)
            ##if i >= data_token[0] and i <= data_token[1]:
                #dev_output_file.write(str(id_to_token[data_sentence[i]])+"\t"+str(id_to_tag[map_dict[pred.item()]])+"\t"+str(id_to_tag[data_label])+"\n")
            #    print(str(id_to_tag[map_dict[label[i]]]))
            #    print(str(id_to_tag[map_dict[pred[i].item()]]))
            
            '''print(i)
            print(str(id_to_token[data_sentence[i]]))
            print(label[i])
            print(map_dict[label[i]])
            #print(str(id_to_tag[map_dict[label[i]]]))
            print(str(id_to_tag[label[i]]))
            print(str(id_to_tag[pred[i].item()]))
            print(str(pred_score[i]))'''
            #dev_output_file.write(str(id_to_token[data_sentence[i]])+"\t"+str(id_to_tag[map_dict[label[i]]])+"\t"+str(id_to_tag[map_dict[pred[i].item()]])+"\t"+str(pred_score[i])+"\t"+str(data_token)+"\n")
            dev_output_file.write(str(id_to_token[data_sentence[i]])+"\t"+str(id_to_tag[label[i]])+"\t"+str(id_to_tag[pred[i].item()])+"\t"+str(pred_score[i])+"\t"+str(data_token)+"\n")
                
            '''if pred.item() > 0:
            dev_output_file.write(str(id_to_token[data_sentence[i]])+"\t"+str(test_cls)+"\t"+str(test_cls)+"\t"+str(pred)+"\n") 
            else:
            dev_output_file.write(str(id_to_token[data_sentence[i]])+"\t"+str(test_cls)+"\t"+str(id_to_tag[pred.item()])+"\t"+str(pred)+"\n")'''
            ##else:
            ##    dev_output_file.write(str(id_to_token[data_sentence[i]])+"\n")
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

    #print(tags)

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

    #print(segments)
    #print("---")
    if prev_tag != 'O':
        segments.append((start_idx, len(tags), prev_tag))
    #print(segments)
    #print("---")
    '''breaked_segments = []
    for s, e, t in segments:
        sidx = s
        for i in range(s+1, e):
            if i in break_idices:
                breaked_segments.append((sidx, i, t))
                sidx = i
        breaked_segments.append((sidx, e, t))'''
    return segments
                                                                                        
