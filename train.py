import argparse, os
import numpy as np
np.random.seed(0)

LSTM_LAYER = 1
LSTM_HIDDEN = 300
DROPOUT_RATE = 0.5
LEARNING_RATE = 0.001
MAX_SENT_LEN = 50
EPOCHS = 30
N_TRAIN_BATCHES = 100
N_VAL_BATCHES = 50
N_TEST_BATCHES = 50

PAD = "__PAD__"
UNK = "__UNK__"


import torch
torch.manual_seed(0)
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from encoder import Encoder
from utils import pprint, ensure_path, Averager, Timer, count_acc, pad_sentences, euclidean_metric, read_data, simplify_token, save_dev_output
from seq_data import SeqDataset
from samplers import CategoriesSampler


output_file = open("./output.txt", 'w')    

def do_pass(batches, counters, shot, way, query, expressions, train, test, id_to_token=None, id_to_tag=None, test_cls=None):
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
        data_token_shot, data_token_query = data_token[:p], data_token[p:]
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
            save_dev_output(output_file, logits, label, data_label, data_sentence_query, data_token_query, query_shot_lens, id_to_token, id_to_tag)

    if test:
        output_file.close()
    return llog, alog
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='data/ontonotes/')
    parser.add_argument('--shot', type=int, default=10)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=10)
    parser.add_argument('--test_way', type=int, default=2)
    parser.add_argument('--glove-path', type=str, default='./data/glove.6B.100d.txt')
    parser.add_argument('--store-glove', action='store_true', help="store all pretrained Glove vectors")
    parser.add_argument('--save-path', type=str, default='./save/proto-1')
    parser.add_argument('--test-class', type=str, default='I-PERSON')
    parser.add_argument('--val-class', type=str, default='I-FAC')
    args = parser.parse_args()
    pprint(vars(args))

    ensure_path(args.save_path)
    
    fname = args.data_path

    train = read_data(fname+"onto.train.bio")
    dev = read_data(fname+"onto.dev.bio")
    test = read_data(fname+"onto.test.bio")

    id_to_token = [PAD, UNK]
    token_to_id = {PAD: 0, UNK: 1}
    id_to_tag = ["O"]
    tag_to_id = {"O": 0}

    for tokens, tags in train + dev + test:
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
            if tag not in tag_to_id:
                tag_to_id[tag] = len(tag_to_id)
                id_to_tag.append(tag)
                
    print(tag_to_id)

    #Fill in the val classes and test classes
    trainset = SeqDataset(train, token_to_id, id_to_token, tag_to_id, id_to_tag, 'train', args.test_class, args.val_class)
    print(set(trainset.data_label))
    train_sampler = CategoriesSampler(trainset.data_label, trainset.data_sent_id, N_TRAIN_BATCHES, args.train_way, args.shot, args.query)

    valset = SeqDataset(dev, token_to_id, id_to_token, tag_to_id, id_to_tag, 'dev', args.test_class, args.val_class)
    val_sampler = CategoriesSampler(valset.data_label, valset.data_sent_id, N_VAL_BATCHES, args.test_way, args.shot, args.query, True)
    print(set(valset.data_label))
    val_batches = [[valset[idx] for idx in indices] for indices, _ in val_sampler]
    val_counter = [[idx for idx in indices] for _, indices in val_sampler]
    
    testset = SeqDataset(test, token_to_id, id_to_token, tag_to_id, id_to_tag, 'test', args.test_class, args.val_class)
    test_sampler = CategoriesSampler(testset.data_label, testset.data_sent_id, N_TEST_BATCHES, args.test_way, args.shot, args.query, True)
    test_batches = [[testset[idx] for idx in indices] for indices, _ in test_sampler]
    test_counter = [[idx for idx in indices] for _, indices in test_sampler]
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

    model = Encoder(pretrained_list, LSTM_HIDDEN, LSTM_LAYER, DROPOUT_RATE)
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
        train_batches = [[trainset[idx] for idx in indices] for indices, _ in train_sampler]
        train_counter = [[idx for idx in indices] for _, indices in train_sampler]
        #exit()
        trn_loss, trn_acc = do_pass(train_batches, train_counter, args.shot, args.train_way, args.query, expressions, True, False, args.test_class)
        trn_loss = trn_loss.item()
        trn_acc = trn_acc.item()

        model.eval()
        _, val_acc = do_pass(val_batches, val_counter, args.shot, args.test_way, args.query, expressions, False, False, id_to_token, id_to_tag, args.test_class)
        val_acc = val_acc.item()
        print('epoch {}, train, loss={:.4f} acc={:.4f}, val acc={:.4f}'.format(epoch, trn_loss, trn_acc, val_acc))

        if val_acc > trlog['max_acc']:
            trlog['max_acc'] = val_acc
            _, tst_acc = do_pass(test_batches, test_counter, args.shot, args.test_way, args.query, expressions, False, True, id_to_token, id_to_tag, args.test_class)
            test_acc = tst_acc.item()
            torch.save({
                    'pretrained_list': pretrained_list,
                    'token_to_id': token_to_id,
                    'model_state_dict': model.state_dict()},
                       os.path.join(args.save_path, 'best.model'))

        trlog['train_loss'].append(trn_loss)
        trlog['train_acc'].append(trn_acc)
        trlog['val_acc'].append(val_acc)
        
    trlog['test_acc'] = test_acc
    print('Final results for test class\t' + args.test_class + ' \t validation class\t'+args.val_class)
    print('Final results, val acc={:.4f}, test acc={:.4f}'.format(trlog['max_acc'], test_acc))
    torch.save(trlog, os.path.join(args.save_path, 'trlog'))

if __name__ == '__main__':
    main()
