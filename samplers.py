import numpy as np
np.random.seed(0)

class CategoriesSampler():
    
    def __init__(self, labels, sent_id, n_batch, n_cls, n_shot, n_query, test=False):
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

        self.classes = list(set(labels))
        self.sent_id = sent_id

        #print(list(set(self.sent_id)))
        #exit()
        
        labels = np.array(labels)
        self.cls_indices = {}
        self.max_ins = -1
        for c in self.classes:
            indices = np.argwhere(labels == c).reshape(-1)
            if c != 0:
                self.max_ins = max(self.max_ins, len(indices))
            self.cls_indices[c] = indices
        self.test = test
        self.max_ins = min(self.max_ins, 300)
    
    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for _ in range(self.n_batch):
            batch = []
            count = []
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
            for c in classes:
                #print("Class-----"+str(c))
                indices = self.cls_indices[c]
                if c == 0:
                    indices = indices[:int(len(indices)/100)]
                #if len(indices) >= self.n_ins:
                permuted_indices = np.random.permutation(indices)
                curr_sent_list = []
                count_instances = 0
                minibatch = []
                minicount = []
                for ind in permuted_indices:
                    if (self.sent_id[ind] not in curr_sent_list or count_instances < self.n_shot) and (count_instances < self.n_ins or self.test==True) and (count_instances < self.max_ins):
                        minibatch.append(ind)
                        minicount.append(1)
                        #print(ind)
                        count_instances += 1
                        if count_instances < self.n_shot:
                            curr_sent_list.append(self.sent_id[ind])
                        #elif count_instances < self.n_ins:
                        #print(str(ind)+" already present")
                if count_instances < self.n_ins:
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
                    #I don't like this because now we are inflating the test set with the same examples over and over again!!!
                    
                batch.append(minibatch)
                count.append(minicount)

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
                
            #print(batch)
            batch = np.stack(batch).flatten('F')
            count = np.stack(count).flatten('F')
             
            yield batch, count
