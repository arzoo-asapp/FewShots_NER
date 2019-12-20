import numpy as np
np.random.seed(0)

from utils import intersection

class CategoriesSampler_Train():
    
    def __init__(self, labels, sent_id, sent_id_dict, n_batch, n_cls, n_shot, n_query, test=False):
        '''
        Args:
            labels: size=(dataset_size), label indices of instances in a data set
            n_batch: int, number of batches for episode training
            n_cls: int, number of sampled classes
            n_ins: int, number of instances considered for a class
        '''
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_ins = n_shot + n_query
        self.n_shot = n_shot
        self.n_query = n_query

        self.classes = list(set(labels))
        self.sent_id = sent_id
        self.labels = labels
        
        labels = np.array(labels)
        self.cls_indices = {}
        self.cls_indices_shot = {}
        self.cls_indices_query = {}
        self.max_ins = -1


        #preprocess sent_id to give the list of all the indices for a given sent_id
        '''sent_ind_dict = dict()
        
        sent_id_list = list(set(list(sent_id)))
        for id_val in sent_id_list:
            print(id_val)
            indices = np.argwhere(np.array(self.sent_id) == id_val).reshape(-1)
            sent_ind_dict[id_val] = indices
        self.sent_ind_dict = sent_ind_dict'''

        self.sent_ind_dict = sent_id_dict

        self.max_sent_id = -1
        for c in self.classes:
            indices = np.argwhere(labels == c).reshape(-1)            
            #self.max_sent_id = max(self.max_sent_id, sent_id[indices[299]])
        
            #for c in self.classes:
            #indices = np.argwhere(labels == c).reshape(-1)
            #indices_query = np.argwhere(np.array(self.sent_id) <= self.max_sent_id).reshape(-1)
            #indices_shot = np.argwhere(np.array(self.sent_id) > self.max_sent_id).reshape(-1)
            
            #if c != 0:
            #    self.max_ins = max(self.max_ins, len(indices))
            self.cls_indices[c] = indices
            #intersection of the indices and indices_shot
            #self.cls_indices_shot[c] = intersection(indices, indices_shot)
            #self.cls_indices_query[c] = intersection(indices, indices_query)

            '''if c != 0:
                print(indices_query)
                print(len(indices_query))
                print(indices)
                print(len(indices))
                print(self.cls_indices_query[c])
                print(len(self.cls_indices_query[c]))
                print(len(self.cls_indices_query[0]))
                exit()'''
            #print(self.cls_indices_shot)
            #print(self.cls_indices_query)

            # if c != 0:
            #self.max_query_ins = max(self.max_ins, len(self.cls_indices_query))                  
            
        self.test = test
        #self.max_ins = min(self.max_ins, 300)
    
    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for x in range(self.n_batch):
            #print("Batch number\t"+str(x))
            batch_shot = []
            batch_query = []
            #batch_labels_query = []
            #Always include "O"? O just means that it does not belong to any group!! And also 'O' 
            #print(self.classes)
            #print(0 in self.classes)
            #new_classes = self.classes.remove(0)
            #print(self.classes)
            if 0 in self.classes: self.classes.remove(0)
            classes = np.random.permutation(self.classes)[:self.n_cls-1]
            classes = np.concatenate((np.array([0]), classes))
            '''
            One possibility is to include the O's in the same sentence as the sentences that have been sampled!!
            '''
            
            batch_labels_query = dict()
            #for c in self.classes + [0]:
            #    batch_labels_query[c] = 0
            for c in range(23):
                batch_labels_query[c] = 0
            
            for new_class, c in enumerate(classes):
                #print("Class-----"+str(c))
                if c == 0:
                    continue
                
                indices = self.cls_indices[c]
                #if c == 0:
                #    indices = indices[:int(len(indices)/100)]
                #if len(indices) >= self.n_ins:
                permuted_indices = np.random.permutation(indices)
                curr_sent_list = []
                count_instances = 0
                minibatch_shot = []
                minibatch_query = []
                #minibatch_labels_query = []

                for ind in permuted_indices:
                    #if (self.sent_id[ind] not in curr_sent_list or count_instances < self.n_shot) and (count_instances < self.n_ins or self.test==True) and (count_instances < self.max_ins):
                    if count_instances < self.n_shot:
                        minibatch_shot.append(ind)
                        #print(ind)
                        count_instances += 1
                        if count_instances < self.n_shot and self.sent_id[ind] not in curr_sent_list:
                            curr_sent_list.append(self.sent_id[ind])
                        #elif count_instances < self.n_ins:
                        #print(str(ind)+" already present")
                    elif count_instances < self.n_ins:
                        count_instances += 1

                        if self.sent_id[ind] not in curr_sent_list:
                            minibatch_query.append(ind)
                            batch_labels_query[c] = new_class

                            sent_id_added = self.sent_id[ind]
                            #print(ind)
                            #print(sent_id_added)
                            #indices_query_zero = np.argwhere(np.array(self.sent_id) == sent_id_added).reshape(-1)
                            ######indices_query_zero = np.random.permutation(self.sent_ind_dict[sent_id_added])[:2]
                            #print(indices_query_zero)
                            #indices_zero = np.argwhere(np.array(self.labels) == 0).reshape(-1)
                            #zero_indices_query = intersection(indices_zero, indices_query_zero)             

                        #for ind in zero_indices_query:
                        '''for ind in indices_query_zero:
                            #print(n_query)
                            #print(n_cls)
                            #print(int((float(self.n_query)/ self.n_cls -1)))
                            if self.labels[ind] == 0 and batch_labels_query.count(0)+minibatch_labels_query.count(0) < (self.n_query):
                                minibatch_query.append(ind)
                                minibatch_labels_query.append(0)'''
                    
                '''if count_instances < self.n_ins:
                    #print("Overflown")
                    #minibatch.append(np.asarray([indices[0]]*(self.n_ins - len(indices))))
                    #minicount.append(np.asarray([0]*(self.n_ins - len(indices))))
                    minibatch += [indices[0]]*(self.n_ins - count_instances)
                    minicount += [0]*(self.n_ins - count_instances)                 
                    
                if count_instances < self.max_ins and self.test==True:
                    #minibatch.append(np.asarray([indices[0]]*(self.max_ins - count_instances)))
                    minibatch += [indices[0]]*(self.max_ins - count_instances)
                    #minicount.append(np.asarray([0]*(self.max_ins - count_instances)))
                    minicount += [0]*(self.max_ins - count_instances)
                    #I don't like this because now we are inflating the test set with the same examples over and over again!!!'''
                    
                batch_shot.append(minibatch_shot)
                #count.append(minicount)
                batch_query += minibatch_query
                #batch_labels_query+= minibatch_labels_query

                '''if(len(batch_query) != len(batch_labels_query)):
                    print("Size mismatch")
                    print(str(len(batch_query))+"\t"+str(len(batch_labels_query)))'''

                #print(len(minibatch))
                
                #batch.append(np.random.permutation(indices)[:self.n_ins])
                '''else:
                    #print(c)
                    #print(np.asarray([indices[0]]*(self.n_ins - len(indices))))
                    temp_arr = np.asarray([indices[0]]*(self.n_ins - len(indices)))
                    #print(len(indices))
                    rand_perm = np.random.permutation(indices)
                    #print(rand_perm)
                    concat_arrays = np.concatenate((rand_perm, temp_arr))
                    batch.append(np.concatenate((np.random.permutation(indices), np.asarray([indices[0]]*(self.n_ins - len(indices))))))
                    #batch.append([0]*(self.n_ins - len(indices)))'''
                
            batch_shot = np.stack(batch_shot).flatten('F')
            
            #if len(batch_query) != len(batch_labels_query):
            #print(len(batch_query))
            #print(len(batch_labels_query))
            #print("---")                                    
            
            yield batch_shot, batch_query, batch_labels_query
