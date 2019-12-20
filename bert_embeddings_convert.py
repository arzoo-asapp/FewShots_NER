#from allennlp.commands.elmo import ElmoEmbedder

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

import logging
logging.basicConfig(level=logging.INFO)

import unidecode
import numpy as np

print("Dev started here!!!")

path="./data/ontonotes/"

#elmo = ElmoEmbedder()
tokens = ["I", "ate", "an", "apple", "for", "breakfast"]

'''vectors = elmo.embed_sentence(tokens)                                                                                                
print(len(vectors[2]))
print(len(vectors[1]))
print(len(vectors[0]))
exit()'''

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = BertModel.from_pretrained('bert-base-uncased')
model.eval()
model.to('cuda')

bert_embeddings = open(path+"bert_onto.nw.test.bio", "w+")

all_sents = open(path+"onto.nw.test.bio")
#all_sents = open(path+"anno_ACE05_BILOU_POS_Split.txt")                                                                          
current_sent = ""
current_sent_vect = []
count_sent = 0
for word in all_sents :
    if len(word.split("\t")) <= 1 :
        #end of sentence                                                                                                          
        count_sent += 1
        #print count_sent                                                                                                         
        #print all_sent_ids_dict[count_sent]                                                                                      
        #print train_ids                                                                                                          
        '''if all_sent_ids_dict[count_sent] in train_ids :
            #print current_sent                                                                                                   
            train_file.write(current_sent+"\n")'''

        #current_sent.insert(0, '[CLS]')
        #current_sent.append('[SEP]')
        current_sent = '[CLS] '+ current_sent + "[SEP]"
        
        tokenized_text = tokenizer.tokenize(current_sent)
        print(tokenized_text)
        
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0]*len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        #model = BertModel.from_pretrained('bert-base-uncased')
        #model.eval()

        tokens_tensor = tokens_tensor.to('cuda')
        segments_tensors = segments_tensors.to('cuda')
        #model.to('cuda')
        
        # Predict hidden states features for each layer                                                                                           
        with torch.no_grad():
            encoded_layers, _ = model(tokens_tensor, segments_tensors)
        # We have a hidden states for each of the 12 layers in model bert-base-uncased
        #last hidden state for all of them!!
        
        vectors = encoded_layers[-4][0] + encoded_layers[-3][0] + encoded_layers[-2][0] + encoded_layers[-1][0]
        vectors = vectors/4.0

        #vectors = np.array([0.]*768)
        #print(vectors)
        #print(vectors.size())

        k = 0
        #vectors = elmo.embed_sentence(current_sent)
        for i in range(len(current_sent_vect)):
            #print(current_sent_vect[i]+"---"+str(current_sent_vect[i]=="anyway")+"~~"+str(current_sent_vect[i]==current_sent_vect[i]))
            #if current_sent_vect[i] == 'anyway':
            #    print("True")
            #    current_sent_vect_word = current_sent_vect[i]
            #else:
            #    print("False")
            current_sent_vect_word = unidecode.unidecode(current_sent_vect[i])

            if current_sent_vect_word == "ianyway":
                current_sent_vect_word = "anyway"
            
            #print(current_sent_vect_word+"\t"+str(current_sent_vect_word == "ianyway"))
            bert_embeddings.write(str(current_sent_vect[i].strip())+"\t")
            k += 1
            current_token = tokenized_text[k]
            current_token = unidecode.unidecode(current_token)
            current_token.replace("##", '')
            print("----")
            print(k)
            print(current_token)
            print(current_sent_vect[i])
            print(current_sent_vect_word)
            while (k < len(tokenized_text)-1 and current_sent_vect_word != current_token and current_token in current_sent_vect_word):
            #while (k < len(tokenized_text)-1 and len(current_sent_vect_word != )):
                k += 1
                current_token += tokenized_text[k].replace("##", '')
                print(current_token)
                if current_token == current_sent_vect_word:
                    print(current_token)
                    break
            if current_token != current_sent_vect_word:
                print("Did not match")
                print(current_token)
                print(current_sent_vect_word)
                
                exit()
                
            for j in vectors[k]:
                #bert_embeddings.write(str(vectors[0][i][j])+"\t")
                bert_embeddings.write(str(j.item())+"\t")
            bert_embeddings.write("\n")
        bert_embeddings.write("\n")
        #exit()
        current_sent = ""
        current_sent_vect = []
    else :
        #current_sent += word.lower()                                                                                             
        current_sent_vect.append(word.split("\t")[0].lower())
        current_sent = current_sent+(word.split("\t")[0].lower())+" "
        #print current_sent                                                                                                       

print(count_sent)        
bert_embeddings.close()
        
        
#vectors = elmo.embed_sentence(tokens)
#print (vectors[2][0])

'''from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, 2, dropout=0)'''
