import argparse, os
import numpy as np
np.random.seed(0)

import operator
import sys
np.set_printoptions(threshold=sys.maxsize)

LSTM_LAYER = 1
LSTM_HIDDEN = 100
DROPOUT_RATE = 0.5
LEARNING_RATE = 0.001
MAX_SENT_LEN = 50
EPOCHS = 50
N_TRAIN_BATCHES = 200
N_VAL_BATCHES = 20
N_TEST_BATCHES = 20

PAD = "__PAD__"
UNK = "__UNK__"


import torch
torch.manual_seed(0)
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from span_encoder import Span_Encoder
from utils import pprint, ensure_path, Averager, Timer, count_acc, pad_sentences, pad_query_sentences, euclidean_metric, read_data, read_bert_data, simplify_token, save_dev_span_output, count_F, count_F_old, Aggregate_F
from span_data import SpanDataset
from samplers import CategoriesSampler
from samplers_test import CategoriesSampler_Test
from samplers_train import CategoriesSampler_Train

#output_file = open("./output.txt", 'w')


def segments_labels(data_sentence_labels_shot_old, batch_labels, id_to_tag, lengths=5):
    interval_sent_batch = []
    interval_label_sent_batch = []
    for i, sent in enumerate(data_sentence_labels_shot_old):
        original_labels_sent = []
        interval_sent = []
        interval_label_sent = []
        for j, token in enumerate(sent):
            if batch_labels[token] !=0:
                original_labels_sent.append("I-"+id_to_tag[token])
            else:
                original_labels_sent.append('O')
            for k in range(j+1):
                if j+1-k < lengths:
                    interval_sent.append([i, k, j])
                    #original_labels.append(original_labels_sent)
        #print(original_labels_sent)
        gold_segments, gold_labels = (get_segments(original_labels_sent))
        #print(gold_segments)
        #print(gold_labels)
        gold_labels_added = [False]*len(gold_labels)

        for [a, b, c] in interval_sent:
            interval = [b, c]
            #print(interval)
            if interval in gold_segments:
                #print("Present")
                interval_label_sent.append(gold_labels[gold_segments.index(interval)])
                gold_labels_added[gold_segments.index(interval)] = True
            else:
                interval_label_sent.append("O")
                #print("---")
                #print(interval_label_sent)
        for p, added in enumerate(gold_labels_added):
            #print("Added long intervals")
            if added == False:
                #print("Added long intervals")
                [b, c] = gold_segments[p]
                interval_sent.append([i, b, c])
                interval_label_sent.append(gold_labels[p])
                #print(interval_label_sent)
                
        interval_sent_batch += interval_sent
        interval_label_sent_batch += interval_label_sent
    return interval_sent_batch, interval_label_sent_batch

        

def do_train_pass(train_batches_shot, train_batches_query, train_batches_labels, shot, way, query, expressions, train, test, id_to_token=None, id_to_tag=None, tag_to_id=None, test_cls=None):
    model, optimizer = expressions
    llog, alog = Averager(), Averager()
            
    for i, (batch_shot, batch_query, batch_labels) in enumerate(zip(train_batches_shot, train_batches_query, train_batches_labels), 1):
        
        flog = Aggregate_F()
        flog_old = Aggregate_F()
        data_token_shot = [x for _, _, x, _, _, _, _ in batch_shot]
        data_sentence_shot = [sent[sent_id] for sent, _, _, _, _, sent_id, _ in batch_shot]
        data_sentence_labels_shot = [label[sent_id] for _, label, _, _, _, sent_id, _ in batch_shot]
        data_sentence_bert_shot = [bert_emb[sent_id] for _, _, _, bert_emb, _, sent_id, _ in batch_shot]


        data_sentence_labels_shot = [[int(batch_labels[token]) for token in sent] for sent in data_sentence_labels_shot]
                
        #print(data_sentence_labels_shot)

        (data_sentence_shot, data_sentence_labels_shot, data_sentence_bert_shot, sentence_shot_lens) = pad_query_sentences(data_sentence_shot, data_sentence_labels_shot, data_sentence_bert_shot, MAX_SENT_LEN, PAD_CLS=way+1)

        #data_sentence_labels_shot = [[int(batch_labels[token]) for token in sent] for sent in data_sentence_labels_shot]
        #print(data_sentence_labels_shot)
        #exit()
        
        proto = model(data_sentence_shot, data_token_shot, data_sentence_bert_shot, sentence_shot_lens, shot=True)

        sorted_batch_labels = sorted(batch_labels.items(), key= lambda kv: (kv[1], kv[0]))
        ###print(sorted_batch_labels)

        #start with the zero!!
        zero_indices = np.argwhere(data_sentence_labels_shot == 0)
        ###print(zero_indices)
        ###print(zero_indices.size())
        old_proto = proto[zero_indices[0], zero_indices[1]]
        ###print(proto.size())
        ###print(old_proto.size())
        new_proto = model.return_attn()(old_proto)
        #print(model.return_attn())
        #print(new_proto.size())
        weights = F.softmax(new_proto, dim=0)
        #print(weights)
        #print(weights.size())
        ###print(weights.size())
        new_proto = (weights * old_proto).sum(dim=0, keepdim=True)
        #print(new_proto)
        ###print(new_proto.size())
        # exit()
        #print(sorted_batch_labels)
        
        #Aggregate all the values of the same label and then take mean!!
        for (key, val) in sorted_batch_labels:
            if val != 0:
                val_indices = np.argwhere(data_sentence_labels_shot == val)
                #print(val)
                #print(val_indices.size())
                new_proto = torch.cat([new_proto, proto[val_indices[0], val_indices[1]].mean(dim=0, keepdim=True)], dim=0)

        ###print(new_proto.size())
        ###exit()
                
        ###proto = proto.reshape(shot, way-1, -1).mean(dim=0)
        
        ###dim_size = proto.size()[1]
        
        ###proto = torch.cat([torch.zeros(1, dim_size).to(device), proto])

        data_token_query = [x for _, _, x, _, _, _, _ in batch_query]
        data_sentence_query = [sent[sent_id] for sent, _,  _, _, _, sent_id, _ in batch_query]
        data_sentence_labels_query = [label[sent_id] for _, label, _, _, _,  sent_id, _ in batch_query]
        data_sentence_bert_query = [bert_emb[sent_id] for _, _, _, bert_emb,  _, sent_id, _ in batch_query]
        
        '''zero_indices = np.argwhere(np.array(batch_labels) == 0)        
        nonzero_indices_part = np.argwhere(np.array(batch_labels) > 0)
        nonzero_indices = []

        for i in range(int(len(zero_indices)/ float(query))):
            nonzero_indices += [ind[0] for ind in nonzero_indices_part]'''
            
        #zero_indices = np.argwhere(np.array(batch_labels) == 0)               
        #batch_labels = [batch_labels[ind] for ind in nonzero_indices] + [batch_labels[ind[0]] for ind in zero_indices]
        #data_token_query = [data_token_query[ind] for ind in nonzero_indices] + [data_token_query[ind[0]] for ind in zero_indices]
        #data_sentence_query = [data_sentence_query[ind] for ind in nonzero_indices] + [data_sentence_query[ind[0]] for ind in zero_indices]                        
        ##batch_labels = [label-1 for label in batch_labels]

        '''count6 = 0
        count7 = 0
        for sent_label in data_sentence_labels_query:
            for token in sent_label:
                if token == 6 :
                    count6+=1
                if token == 7:
                    count7+=1
        print("Count of 6\t"+str(count6)+"\tCount of 7\t"+str(count7))'''
                    
        data_sentence_labels_query = [[int(batch_labels[token]) for token in sent] for sent in data_sentence_labels_query]
        #print(batch_labels)
        
        #print(np.argwhere(data_sentence_labels_query == np.array(batch_labels).any()))
        
        #labels = torch.LongTensor(np.array(batch_labels)).to(device)

        (data_sentence_query, labels, data_sentence_bert_query, sentence_query_lens) = pad_query_sentences(data_sentence_query, data_sentence_labels_query, data_sentence_bert_query, MAX_SENT_LEN, PAD_CLS = way+1)

        query_matrix = model(data_sentence_query, data_token_query, data_sentence_bert_query, sentence_query_lens)

        for vec, index, sentence in zip(query_matrix, data_token_query, data_sentence_query):
            if vec.sum() == 0.:
                print(index[0])
                for token in sentence:
                    token = token.to('cpu').item()
                    if token == "__PAD__":
                        continue
                    print(token)
                    #print(id_to_token)
                    print(id_to_token[int(token)])
                print("Finally-------------------------")
                print(id_to_token[sentence[index[0]].to('cpu').item()])
                print("---")
        
        logits = euclidean_metric(query_matrix, new_proto)
        #print("Training_logits\t")
        #print(logits)
        #print(logits.size())
        
        logits[:, :, 0] = model.return_0class()
        ###print(logits)
        
        softmax_scores = F.softmax(logits, dim=2)
        #print(softmax_scores)

        #labels = torch.LongTensor(np.array(data_sentence_labels_query)).to(device)
        logits_t = logits.transpose(2, 1)
        #print(logits_t.size())
        #print(labels.size())
        #exit()

        loss_function = torch.nn.CrossEntropyLoss(ignore_index=way+1)
        loss = loss_function(logits_t, labels)
        #loss = F.cross_entropy(logits_t, labels)
        llog.add(loss.item())            
        
        correct, total_preds, total_gold, confidence = count_F(softmax_scores, labels, batch_labels.values(), train=True, PAD_CLS=way+1, id_to_tag=id_to_tag)
        
        flog.add(correct, total_preds, total_gold)
        item1, item2, item3 = flog.item()
        print("Correct")
        print(item1)
        print("Predicted")
        print(item2)
        print("Gold")
        print(item3)

        f_score1 = flog.f_score()
        print(f_score1)
        
        print("Token-level accuracy----")
        correct, total_preds, total_gold, confidence = count_F_old(softmax_scores, labels, batch_labels.values(), train=True, PAD_CLS=way+1)

        flog_old.add(correct, total_preds, total_gold)
        item1, item2, item3 = flog_old.item()
        print("Correct")
        print(item1)
        print("Predicted")
        print(item2)
        print("Gold")
        print(item3)
        
        f_score1_old = flog_old.f_score()
        print(f_score1_old)

        #exit()
        
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                                                
    return llog, flog


        
def do_test_pass(test_batches_shot, test_batches_query, test_batches_labels, shot, way, query, expressions, train, test, id_to_token=None, id_to_tag=None, test_cls=None):
    model, optimizer = expressions
    llog, alog = Averager(), Averager()
    
    if test:
        output_file = open("./output.txt"+str(test_cls), 'w')

    for i, (batch_shot, batch_query, batch_labels) in enumerate(zip(test_batches_shot, test_batches_query, test_batches_labels), 1):

        #print(batch_labels.keys())

        #batch_query = batch_shot
        
        data_token_shot = [x for _, _, x, _, _, _, _ in batch_shot]
        data_sentence_shot = [sent[sent_ind] for sent, _, _, _, _, sent_ind, _ in batch_shot]
        data_sentence_labels_shot = [label[sent_ind] for _, label, _, _, _, sent_ind, _ in batch_shot]
        data_sentence_bert_shot = [bert_emb[sent_ind] for _, _, _, bert_emb, _, sent_ind, _ in batch_shot]
                
        #(data_sentence_shot, sentence_shot_lens)= pad_sentences(data_sentence_shot, MAX_SENT_LEN)

        data_sentence_labels_shot = [[int(batch_labels[token]) for token in sent] for sent in data_sentence_labels_shot]
                
        (data_sentence_shot, data_sentence_labels_shot, data_sentence_bert_shot, sentence_shot_lens) = pad_query_sentences(data_sentence_shot, data_sentence_labels_shot, data_sentence_bert_shot, MAX_SENT_LEN, PAD_CLS=way+1)
                
        sorted_batch_labels = sorted(batch_labels.items(), key= lambda kv: (kv[1], kv[0]))

        zero_indices = np.argwhere(data_sentence_labels_shot == 0)
        
        proto = model(data_sentence_shot, data_token_shot, data_sentence_bert_shot, sentence_shot_lens, shot=True)
        
        old_proto = proto[zero_indices[0], zero_indices[1]]
                
        new_proto = model.return_attn()(old_proto)
        weights = F.softmax(new_proto, dim=0)
        new_proto = (weights * old_proto).sum(dim=0, keepdim=True)
                
        for (key, val) in sorted_batch_labels:
            if val != 0:
                val_indices = np.argwhere(data_sentence_labels_shot == val)
                #print(val)
                #print(val_indices.size())
                new_proto = torch.cat([new_proto, proto[val_indices[0], val_indices[1]].mean(dim=0, keepdim=True)], dim=0)
                                                                                                    
        
        #proto = model(data_sentence_shot, data_token_shot, sentence_shot_lens, shot=True)
        #proto = proto.reshape(shot, way-1, -1).mean(dim=0)

        #dim_size = proto.size()[1]

        #proto = torch.cat([torch.zeros(1, dim_size).to(device), proto])

        #batch_query = batch_shot
        
        data_token_query = [x for _, _, x, _, _, _, _ in batch_query]
        data_sentence_query = [sent[ind] for sent, _,  _, _, _, ind, _ in batch_query]
        data_sentence_labels_query = [label[ind] for _, label, _, _, _, ind, _ in batch_query]
        data_sentence_bert_query = [bert_emb[ind] for _, _, _, bert_emb, _, ind, _ in batch_query]

        data_sentence_labels_query = [[int(batch_labels[token]) for token in sent] for sent in data_sentence_labels_query]

        #print(len(data_sentence_query))
        #print(len(data_sentence_labels_query))
        
        #batch_index = 0
        block_size = 1000

        flog = Aggregate_F()
        flog_old = Aggregate_F()
        #confidence = []
        
        for batch_index in np.arange(0, len(data_sentence_labels_query), block_size):
            #print(batch_index)
            #if(batch_index > 0):
            #    continue
            ##exit()
            mini_data_token_query = data_token_query[batch_index: batch_index+block_size]
            mini_data_sentence_query = data_sentence_query[batch_index: batch_index+block_size]
            mini_data_sentences_labels_query = data_sentence_labels_query[batch_index: batch_index+block_size]
            mini_data_sentence_bert_query = data_sentence_bert_query[batch_index: batch_index+block_size]

            #print(mini_data_token_query.size())
            #print(mini_sentence_query.size())
            #print(mini_data_sentences)
            
            (mini_data_sentence_query, labels, mini_data_sentence_bert_query, mini_sentence_query_lens)= pad_query_sentences(mini_data_sentence_query, mini_data_sentences_labels_query, mini_data_sentence_bert_query, MAX_SENT_LEN, PAD_CLS=way+1)

            '''class_indices = np.argwhere(np.array(batch_labels) > 0)
            new_batch_labels = np.array(batch_labels)
            new_batch_labels[class_indices] = 1
                                                
        
            old_mini_labels = batch_labels[batch_index: batch_index+block_size]'''
            
            #print(len(labels))
            #print(old_mini_labels)
            '''class_indices = np.argwhere(np.array(old_mini_labels) > 0)
            mini_labels = np.array(old_mini_labels)
            mini_labels[class_indices] = 1'''
            #print(mini_labels)
            #print(set(batch_labels))
            #print("--")
            #mini_labels = torch.LongTensor(labels).to(device)
            #print(labels)
            mini_labels = labels
            
            logits = euclidean_metric(model(mini_data_sentence_query, mini_data_token_query, mini_data_sentence_bert_query, mini_sentence_query_lens), new_proto)
            ###print("Test_logits\t")
            ###print(logits)
                            
            logits[:, :, 0] = model.return_0class()
            ###print(logits)
            
            softmax_scores = F.softmax(logits, dim=2)
            #print(softmax_scores)
            
            ##loss = F.cross_entropy(logits, labels)
            ##acc = count_acc(logits, mini_labels)
            ##llog.add(loss.item())
            ##alog.add(acc)
            if test:
                #print the outputs to a file
                save_dev_span_output(output_file, softmax_scores, np.array(mini_labels), batch_labels, mini_data_sentence_query, mini_data_token_query, mini_sentence_query_lens, id_to_token, id_to_tag, test_cls)
            correct, total_preds, total_gold, mini_confidence = count_F(softmax_scores, mini_labels, batch_labels.values(), train=False, id_to_tag=id_to_tag, PAD_CLS=way+1)

            #confidence += mini_confidence

            flog.add(correct, total_preds, total_gold)
            item1, item2, item3 = flog.item()
            #print(item1)
            #print(item2)
            #print(item3)


            correct, total_preds, total_gold, mini_confidence = count_F_old(softmax_scores, mini_labels, batch_labels.values(), train=False, PAD_CLS=way+1)

            #confidence += mini_confidence
            
            flog_old.add(correct, total_preds, total_gold)
            #item1, item2, item3 = flog.item()

        f_score1 = flog.f_score()
        print(f_score1)
        item1, item2, item3 = flog.item()
        print("Correct")
        print(item1)
        print("Predicted")
        print(item2)
        print("Gold")
        print(item3)

        print("Token level accuracy-----")

        f_score1 = flog_old.f_score()
        print(f_score1)
        item1, item2, item3 = flog_old.item()
        print("Correct")
        print(item1)
        print("Predicted")
        print(item2)
        print("Gold")
        print(item3)
                                                        
        #confidence.sort(key=lambda pair: pair[0])
        #confidence.sort(key=lambda pair: pair[1])
        #print(confidence)
        
        #exit()
    if test:
        output_file.close()
    return flog_old, flog

        

'''def do_pass(batches, counters, shot, way, query, expressions, train, test, id_to_token=None, id_to_tag=None, test_cls=None):
    model, optimizer = expressions
    llog, alog = Averager(), Averager()

    if test:
        output_file = open("./output.txt"+str(test_cls), 'w')
    
    for i, (batch, counter) in enumerate(zip(batches, counters), 1):
        #print("Batch number\t"+str(i))
        data_token = [x for _, x, _, _ in batch]
        data_sentence = [sent for sent, _, _, _ in batch]
        data_label = [label for _, _, label, _ in batch]
        p = shot * way
        #print(len(data_token))
        #print(p)
        #print(shot)
        #print(way)
        data_token_shot, data_token_query = np.array(data_token[:p]), np.array(data_token[p:])
        
        data_sentence_shot, data_sentence_query = data_sentence[:p], data_sentence[p:]
        counter_token, counter_query = counter[:p], counter[p:]
        
        (data_sentence_shot, sentence_shot_lens), (data_sentence_query, query_shot_lens) = pad_sentences(data_sentence_shot, MAX_SENT_LEN), pad_sentences(data_sentence_query, MAX_SENT_LEN)
        
        proto = model(data_sentence_shot, data_token_shot, sentence_shot_lens)
        proto = proto.reshape(shot, way, -1).mean(dim=0)

        ####label = torch.arange(way).repeat(query)
        if not train:
            #print(len(data_token))
            #print(p)
            #print(way)
            query = int((len(data_token) - p)/way)
            #print(query)
            #exit()
        
        label = torch.arange(way).repeat(query)
        label = label.type(torch.LongTensor).to(device)
        
        logits = euclidean_metric(model(data_sentence_query, data_token_query, query_shot_lens), proto)

        #print(list(model.parameters()))
        #print(model.return_0class())

        #print(logits.size())
        logits[:, 0] = model.return_0class()
        #print(logits.size())
        #print(label.size())
        #print(len(counter_query))
        #print(counter_query)
        #print("---")
        
        loss = F.cross_entropy(logits, label)
        acc = count_acc(logits, label, counter_query)

        llog.add(loss.item())
        alog.add(acc)
        
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if test:
            #print the outputs to a file
            save_dev_span_output(output_file, logits, label, data_label, data_sentence_query, data_token_query, query_shot_lens, id_to_token, id_to_tag)

    if test:
        output_file.close()
    return llog, alog'''
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='data/ontonotes/')
    parser.add_argument('--shot', type=int, default=10)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=3)
    parser.add_argument('--test_way', type=int, default=2)
    parser.add_argument('--glove-path', type=str, default='./data/glove.6B.100d.txt')
    parser.add_argument('--store-glove', action='store_true', help="store all pretrained Glove vectors")
    parser.add_argument('--save-path', type=str, default='./save/proto-1')
    parser.add_argument('--test-class', type=str, default='I-PERSON')
    parser.add_argument('--val-class', type=str, default='I-FAC')
    args = parser.parse_args()
    pprint(vars(args))
    print("TEST CLASSS~~~~~~"+str(args.test_class))
    ensure_path(args.save_path)    
    fname = args.data_path

    #train = read_data(fname+"onto.train.bio")
    #dev = read_data(fname+"onto.dev.bio")
    #test = read_data(fname+"onto.test.bio")

    bert_train = read_bert_data(fname+"onto.train.bio", fname+"bert_onto.train.bio")
    bert_dev = read_bert_data(fname+"onto.test.bio", fname+"bert_onto.dev.bio")
    bert_test = read_bert_data(fname+"onto.test.bio", fname+"bert_onto.test.bio")

    print("File read")
    
    id_to_token = [PAD, UNK]
    token_to_id = {PAD: 0, UNK: 1}
    id_to_tag = ["O"]
    tag_to_id = {"O": 0}

    #for tokens, tags, bert_emb in bert_train + bert_dev + bert_test:
    for tokens, tags, bert_emb in bert_train + bert_test:
        #print(tokens)
        #print(tags)
        for token in tokens:
                token = simplify_token(token)
                '''for char in token:
                    if char not in char_to_id:
                        char_to_id[char] = len(char_to_id)
                        id_to_char.append(char)'''
                token = token.lower() # use lowercased tokens but original characters                                
                if token not in token_to_id:
                    token_to_id[token] = len(token_to_id)
                    id_to_token.append(token)
        for tag in tags:
            #print(tag)
            if "B-" in tag or "I-" in tag:
                tag = tag[2:]
            #print(tag)
            if tag not in tag_to_id:
                tag_to_id[tag] = len(tag_to_id)
                #if len(tag_to_id) == 2:
                #print(tag)
                #print(tag_to_id)
                #exit()
                id_to_tag.append(tag)
                
    #print(tag_to_id)
    #exit()
    #Fill in the val classes and test classes
    trainset = SpanDataset(bert_train, token_to_id, id_to_token, tag_to_id, id_to_tag, 'train', args.test_class, args.val_class)
    print(set(trainset.data_label))
    
    train_sampler = CategoriesSampler_Train(trainset.data_label, trainset.data_sent_id, trainset.data_sent_id_dict, N_TRAIN_BATCHES, args.train_way, args.shot, args.query)
    
    valset = SpanDataset(bert_dev, token_to_id, id_to_token, tag_to_id, id_to_tag, 'dev', args.test_class, args.val_class)
    val_sampler = CategoriesSampler_Test(valset.data_label, valset.data_sent_id, valset.data_sent_id_dict, N_VAL_BATCHES, args.test_way, args.shot, args.query, True)
    print(set(valset.data_label))

    val_batches_shot = []
    val_batches_query = []
    val_batches_query_labels = []
    for (shot_indices, query_indices, labels) in val_sampler:
        val_batches_shot.append([valset[idx] for idx in shot_indices])
        val_batches_query.append([valset[idx] for idx in query_indices])
        val_batches_query_labels.append(labels)

        
    '''val_batches_shot = [[valset[idx] for idx in indices] for indices, _, _ in val_sampler]
    val_batches_query = [[valset[idx] for idx in indices] for _, indices, _ in val_sampler]
    val_batches_query_labels = [labels for _, _, labels in val_sampler]'''

    #val_counter = [[idx for idx in indices] for _, indices in val_sampler]
    
    testset = SpanDataset(bert_test, token_to_id, id_to_token, tag_to_id, id_to_tag, 'test', args.test_class, args.val_class)
    test_sampler = CategoriesSampler_Test(testset.data_label, testset.data_sent_id, testset.data_sent_id_dict, N_TEST_BATCHES, args.test_way, args.shot, args.query, True)
    '''test_batches_shot = [[testset[idx] for idx in indices] for indices, _, _ in test_sampler]
    test_batches_query = [[testset[idx] for idx in indices] for _, indices, _ in test_sampler]
    test_batches_query_labels = [labels for _, _, labels in test_sampler]'''

    test_batches_shot = []
    test_batches_query = []
    test_batches_query_labels = []
    for (shot_indices, query_indices, labels) in test_sampler:
        test_batches_shot.append([testset[idx] for idx in shot_indices])
        test_batches_query.append([testset[idx] for idx in query_indices])
        test_batches_query_labels.append(labels)
                                                
    #test_counter = [[idx for idx in indices] for _, indices in test_sampler]
    print(set(testset.data_label))
    #id_to_token, token_to_id = trainset.id_to_token, trainset.token_to_id

    pretrained = {}
    word_emb_size = 0
    for line in open(args.glove_path):
        parts = line.strip().split()
        word = parts[0]
        vector = [float(v) for v in parts[1:]]
        pretrained[word] = vector
        word_emb_size = len(vector)
        
        if args.store_glove and word not in token_to_id:
            token_to_id[word] = len(token_to_id)
            id_to_token.append(word)

    pretrained_list = []
    scale = np.sqrt(3.0 / word_emb_size)
    for word in id_to_token:
        #apply lower() because all glove vectors are for lowercase words
        if word.lower() in pretrained:
            pretrained_list.append(np.array(pretrained[word.lower()]))
        else:
            random_vector = np.random.uniform(-scale, scale, [word_emb_size])
            pretrained_list.append(random_vector)

    model = Span_Encoder(pretrained_list, LSTM_HIDDEN, LSTM_LAYER, DROPOUT_RATE)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    timer = Timer()
    expressions = (model, optimizer)
    test_acc = 0.
    for epoch in range(1, EPOCHS + 1):
        model.train()
        #train_batches = [[trainset[idx] for idx in indices] for indices, _ in train_sampler]
        #train_counter = [[idx for idx in indices] for _, indices in train_sampler]

        train_batches_shot = []
        train_batches_query = []
        train_batches_query_labels = []
        for (shot_indices, query_indices, labels) in train_sampler:
            train_batches_shot.append([trainset[idx] for idx in shot_indices])
            train_batches_query.append([trainset[idx] for idx in query_indices])
            train_batches_query_labels.append(labels)
            
        '''train_batches_shot = [[trainset[idx] for idx in indices] for indices, _, _ in train_sampler]
        train_batches_query = [[trainset[idx] for idx in indices] for _, indices, _ in train_sampler]
        train_batches_query_labels = [labels for _, _, labels in train_sampler]
        testing = [indices for _, indices, _  in train_sampler]

        counter = 0
        global_counter = 0
        for a, b in zip(train_batches_query_labels, testing):
            global_counter += 1
            if len(a) != len(b):
                counter += 1
                print("Unequal\t"+str(counter)+"\t"+str(global_counter))
                print(str(len(a))+"\t"+str(len(b)))
                exit()'''
        
        #exit()
        trn_loss, trn_acc = do_train_pass(train_batches_shot, train_batches_query, train_batches_query_labels, args.shot, args.train_way, args.query, expressions, True, False, id_to_token, id_to_tag, tag_to_id, args.test_class)
        trn_loss = trn_loss.item()
        trn_acc = trn_acc.f_score()

        model.eval()
        
        '''_, val_acc = do_test_pass(val_batches_shot, val_batches_query, val_batches_query_labels, args.shot, args.test_way, args.query, expressions, False, False, id_to_token, id_to_tag, args.val_class)
        #_, val_acc = do_pass(val_batches, val_counter, args.shot, args.test_way, args.query, expressions, False, False, id_to_token, id_to_tag, args.test_class)

        #print(val_acc.f_score().keys())
        val_acc = val_acc.f_score()['DATE']'''
        val_acc = 0.
        print('epoch {}, train, loss={:.4f} val acc={:.4f}'.format(epoch, trn_loss, val_acc))

        #val_acc = 0.
        print("Testing")
        if val_acc >= trlog['max_acc']:
            trlog['max_acc'] = val_acc
            #_, tst_acc = do_pass(test_batches, test_counter, args.shot, args.test_way, args.query, expressions, False, True, id_to_token, id_to_tag, args.test_class)
            _, tst_acc = do_test_pass(test_batches_shot, test_batches_query, test_batches_query_labels, args.shot, args.test_way, args.query, expressions, False, True, id_to_token, id_to_tag, args.test_class)
            
            test_acc = tst_acc.f_score()['DATE']
            torch.save({
                    'pretrained_list': pretrained_list,
                    'token_to_id': token_to_id,
                    'model_state_dict': model.state_dict()},
                       os.path.join(args.save_path, 'best.model'))

        #val_acc = 0.
        #test_acc = 0.
        trlog['train_loss'].append(trn_loss)
        trlog['train_acc'].append(trn_acc)
        trlog['val_acc'].append(val_acc)
        
    trlog['test_acc'] = test_acc
    print('Final results for test class\t' + args.test_class + ' \t validation class\t'+args.val_class)
    print('Final results, val acc={:.4f}, test acc={:.4f}'.format(trlog['max_acc'], test_acc))
    torch.save(trlog, os.path.join(args.save_path, 'trlog'))

if __name__ == '__main__':
    main()
