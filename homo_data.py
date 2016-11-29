import numpy as np
import copy
import utils
class HomogeneousData():
    def __init__(self, index_files, indexKey,batch_size=128,maxlen=None):
        self.batch_size = batch_size
        self.data = utils.load_data_from_disk(index_files[indexKey])
        self.maxlen = maxlen

        self.prepare()
        self.reset()
        print "load {} successfully!, data size {}".format(indexKey, len(self.data))

    def prepare(self):
        #Training Data: [[[word_id_1,word_id_2,....],feature_id],......]
        self.caps = [self.data[i][0] for i in range(len(self.data))]
        self.feats = [self.data[i][1] for i in range(len(self.data))]

        #find the unique lengths
        self.lengths = [len(cc) for cc in self.caps]
        self.len_unique = np.unique(self.lengths)

        if self.maxlen:
            self.len_unique = [ll for ll in self.len_unique if ll <= self.maxlen]
        
        #indices of unique lengths
        self.len_indices = dict()
        self.len_counts = dict()
        for ll in self.len_unique:
            self.len_indices[ll] = np.where(self.lengths == ll)[0]
            self.len_counts[ll] = len(self.len_indices[ll])
        self.len_cur_counts = copy.copy(self.len_counts)

    def reset(self):
        self.len_cur_counts = copy.copy(self.len_counts)
        self.len_unique = np.random.permutation(self.len_unique)
        self.len_indices_pos = {}
        for ll in self.len_unique:
            self.len_indices_pos[ll] = 0
            self.len_indices[ll] = np.random.permutation(self.len_indices[ll])
        self.len_idx = -1

    def next(self):
        count = 0
        while True:
            self.len_idx = np.mod(self.len_idx+1,len(self.len_unique))
            if self.len_cur_counts[self.len_unique[self.len_idx]] > 0:
                break
            count += 1
            if count >= len(self.len_unique):
                break
        if count >= len(self.len_unique):
            self.reset()
            raise StopIteration()

        cur_len = self.len_unique[self.len_idx]
        cur_batch_size = np.minimum(self.batch_size,self.len_cur_counts[cur_len])
        cur_pos = self.len_indices_pos[cur_len]

        cur_indices = self.len_indices[cur_len][cur_pos:cur_pos+cur_batch_size]
        self.len_indices_pos[cur_len] += cur_batch_size
        self.len_cur_counts[cur_len] -= cur_batch_size
        return cur_indices   

    def __iter__(self):
        return self

if __name__ == "__main__":
    print "hello world"
