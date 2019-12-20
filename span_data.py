from collections import Counter
from nltk.tokenize import word_tokenize
import numpy as np

from torch.utils.data import Dataset

from utils import read_data, simplify_token, get_segments

#For the time being, assuming that the label is L which is to be removed from the training set and then 
#only appears in the validation set -- to compute the support set??
#What about the query set??

'''
For the time being, the test class has one label and all the other labels are in the training set...first from the training set set apart a few examples from the class labels as the support set and not train on those for now!!!
'''

PAD = "__PAD__"
UNK = "__UNK__"

class SpanDataset(Dataset):
    
    def __init__(self, file_data, token_to_id, id_to_token, tag_to_id, id_to_tag, setname='train', test_cls={'PERSON', 'PERSON'}, valid_cls={'FAC', 'FAC'}):
        '''
        Args:
        fname: str, training data file
        sent_col: int, the column corresponding to sentences
        tag_col: int, the column corresponding to the entities and labels
        setname: str, dataset name ['train', 'val', 'test']
        n_cls: list of ints, number of classes of the train/val/test sets, these are not overlapping??
        '''
        
        #train_fname = fname+"onto.train.bio"
        #train = read_data(train_fname)
        #dev = read_data(fname+"onto.dev.bio")
        #test = read_data(fname+"onto.test.bio")
        
        #id_to_token = [PAD, UNK]
        #token_to_id = {PAD: 0, UNK: 1}
        #id_to_tag = [PAD]
        #tag_to_id = {PAD: 0}
        #data_token = []
        #data_sentence = []
        #data_label = []

        #fname = fname+"onto."+setname+".bio"
        

        '''with open(fname, 'r') as data_src:
            token_indices, labels = [], []
            for line in data_src:
                if line.startswith("-DOCSTART-"): continue
                parts = line.strip().split()
                if len(parts) < 2:
                    if len(tokens) > 0:
                        #content.append((tokens, tags))
                        for i, (token_index, label) in enumerate(zip(token_indices, labels)):
                            data_sentence.append(token_indices)
                            data_token.append(i)
                            data_label.append(label)
                        token_indices, labels = [], []
                    continue
                token = parts[0]
                tag = parts[-1]
                #tokens.append(parts[0])
                #tags.append(parts[-1])
                if token not in token_to_id:
                    token_to_id[token] = len(token_to_id)
                    id_to_token.append(token)
                token_indices.append(token_to_id[token])
                if tag not in tag_to_id:
                    tag_to_id[tag] = len(tag_to_id)
                    id_to_tag.append(tag)
                #Now check if this is part of the training set or not!!!
                if (setname == 'train' and tag not in test_cls and tag not in valid_cls) \
                        or (setname == 'val' and tag in valid_cls) or \
                        (setname == 'test' and tag in test_cls):
                    labels.append(tag_to_id[tag])
                else:
                    labels.append(tag_to_id['O'])'''

        
        data_sent_id = []
        data_sent_id_dict = dict()
        data_token = []                                                                                              
        data_sentence = []
        data_sentence_label = []
        data_sentence_bert_label = []
        
        data_label = []
        
        for j, data in enumerate(file_data):
            #print(data)
            #for lines in data:
            #    print(lines)
            sentence, tags, bert_embeddings = data

            sent_token_id = []
            counting_spans = 0
            #print(dict_segments)
            #print(len(segments))
            added_sentence = 0

            sent_label = []
            sent_bert_label = []
            
            for i, (token, tag, bert_emb) in enumerate(zip(sentence, tags, bert_embeddings)):
                if "B-" in tag or "I-" in tag:
                    tag = tag[2:]
                sent_token_id.append(token_to_id.get(simplify_token(token).lower(), 1))
                #for k in np.arange(i+1):
                    #data_token.append([token_to_id.get(simplify_token(sentence[k].lower()), 1), token_to_id.get(simplify_token(sentence[i].lower()), 1)])
                #    if i-k > 4:
                #        continue
                counting_spans += 1
                data_token.append([i, i])
                #sent_token_id.append(token_to_id.get(simplify_token(token).lower(), 1))
                data_sent_id.append(j)
                sent_bert_label.append(bert_emb)
                    #Here remove all the tags which are not in the dev set or the test set!!!
                if (setname == 'train' and tag not in test_cls and tag not in valid_cls)\
                        or (setname == 'dev' and tag in valid_cls)\
                        or (setname == 'test' and tag in test_cls):
                    #if str(k)+"|"+str(i) in dict_segments.keys():
                    data_label.append(tag_to_id[tag])
                    #print(j)
                    sent_label.append(tag_to_id[tag])
                    
                    #print(str(tag)+"--"+str(k)+"--"+str(i))
                    #print("---")
                    #num_added += 1
                    added_sentence += 1
                else:
                    data_label.append(tag_to_id["O"])
                    ######Changed here to include all the annotations!!!
                    sent_label.append(tag_to_id[tag])
                #if setname=='dev' and tag in valid_cls:
                #print(i, token, tag)
            
            data_sent_id_dict[j] = np.arange(len(data_sentence)-1, len(data_sentence)-1+1)
            #for token in sentence:
            data_sentence.append(sent_token_id)
            data_sentence_label.append(sent_label)
            data_sentence_bert_label.append(sent_bert_label)
            #print(added_sentence)
            #print("~~~~")
        
        self.id_to_token = id_to_token
        self.token_to_id = token_to_id
        self.id_to_tag = id_to_tag
        self.tag_to_id = tag_to_id
        self.data_sentence = data_sentence
        self.data_sentence_label = data_sentence_label
        self.data_sentence_bert_label = data_sentence_bert_label
        self.data_token = data_token
        self.data_label = data_label

        '''for (l, id1) in zip(data_label, data_sent_id):
            print(str(l)+"\t"+str(id1))
        
        indices = np.argwhere(np.array(data_label) == tag_to_id["ORG"]).reshape(-1)
        print(tag_to_id["ORG"])
        print(indices)
        #print(c)
        #print(indices)
        max_sent_id = data_sent_id[indices[299]]
        print(max_sent_id)'''
        
        self.data_sent_id = data_sent_id
        self.data_sent_id_dict = data_sent_id_dict
        #print(len([x > 0 for x in data_label]))
        #print([x > 0 for x in data_label])
        print("Non zero")
        '''print(num_added)
        print(test_cls)
        print(valid_cls)'''
        print(np.count_nonzero(data_label))
        
    def __len__(self):
        return len(self.data)


    def __getitem__(self, i):
        return self.data_sentence, self.data_sentence_label, self.data_token[i], self.data_sentence_bert_label, self.data_label[i], self.data_sent_id[i], self.data_sent_id_dict
