import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F
import numpy as np
from chainer import cuda
import Queue
import utils
import math

def broadcast_second_axis(v,n):
    #v.shape(batchsize,b) --> (batchsize,n,b)
    batchsize = v.data.shape[0]
    b = v.data.shape[1]
    vt = F.transpose(v,axes=(1,0))
    vt_broadcasted = F.broadcast_to(vt,(n,b,batchsize))
    return F.transpose(vt_broadcasted,axes=(2,0,1))

class MultiNet_init(chainer.ChainList):
    def __init__(self,d_in,d_out,n_layer_init,dropout=None):
        self.n_layer_init = n_layer_init
        self.d_in = d_in 
        self.d_out = d_out
        self.dropout = dropout
        layers = []
        layers.append(L.Linear(d_in,d_out))
        for i in range(n_layer_init-1):
            layers.append(L.Linear(d_out,d_out))
        super(MultiNet_init,self).__init__(*layers)

    def __call__(self,x,train=None):
        h = x
        for layer in self[:-1]:
            if self.dropout is None:
                h = F.relu(layer(h))
            else:
                h = F.dropout(F.relu(layer(h)),ratio=self.dropout,train=train)
        return F.tanh(self[-1](h))

class MultiLNet(chainer.ChainList):
    def __init__(self,d_in,d_out,n_layer_init):
        self.n_layer_init = n_layer_init
        self.d_in = d_in 
        self.d_out = d_out
        layers = []
        layers.append(L.Linear(d_in,d_out))
        for i in range(n_layer_init-1):
            layers.append(L.Linear(d_out,d_out))
        super(MultiLNet,self).__init__(*layers)

    def __call__(self,x):
        h = x
        for layer in self[:-1]:
            h = layer(h)
        #h = F.tanh(self[-1](h))
        h = self[-1](h)
        return h

class Attention(chainer.Chain):
    def __init__(self,hidden_size,context_size,n_layer_att):
        super(Attention,self).__init__(
            hw=L.Linear(hidden_size,context_size),
            cw=MultiLNet(context_size,context_size,n_layer_att),
            we=L.Linear(context_size,1)
            )
        self.hidden_size = hidden_size
        self.context_size = context_size


    def __call__(self,context,h,train=True):
        #context :  bs*length*512
        bs = h.data.shape[0]
        context3d2h = F.reshape(self.cw(F.reshape(context,(-1,self.context_size))),(bs,14*14,self.context_size))
        # (bs,hiddensize) -(bs,14*14,hiddensize)
        h2h = broadcast_second_axis(self.hw(h),14*14)
        weight = F.reshape(self.we(F.reshape(F.relu(context3d2h+h2h),(-1,self.context_size))),(bs,14*14))
        return F.softmax(weight) #(bs,14*14)

    def arctic__call__(self,context,h,train=True):
        #context :  bs*length*512
        bs = h.data.shape[0]
        context3d2h = F.reshape(self.cw(F.reshape(context,(-1,self.context_size))),(bs,14*14,self.context_size))
        # (bs,hiddensize) -(bs,14*14,hiddensize)
        h2h = broadcast_second_axis(self.hw(h),14*14)
        weight = F.reshape(self.we(F.reshape(F.tanh(context3d2h+h2h),(-1,self.context_size))),(bs,14*14))
        return F.softmax(weight) #(bs,14*14)


class Decoder(chainer.Chain):
    def __init__(self,vocab_size,embed_size,context_size,hidden_size,dropout_ratio,index_files):
        super(Decoder,self).__init__(
            ye=L.EmbedID(vocab_size+3,embed_size),  # <eos> 0 , UNK 1, <bos> 2,
            eh=L.Linear(embed_size,4*hidden_size),
            hh=L.Linear(hidden_size,4*hidden_size),
            ch=L.Linear(context_size,4*hidden_size),

            #deep output layer, from lstm to y_t
            hf=L.Linear(hidden_size,embed_size),
            cf=L.Linear(context_size,embed_size),
            L0=L.Linear(embed_size,vocab_size+3)
            )
        self.dropout_ratio = dropout_ratio
        self.dropout_lstm_ratio = dropout_ratio
        print "do not load pretrained vector! "

        """
        wordId_wordEmbed_dict = utils.load_data_from_disk(index_files["id_embed"])
        print "vocab size is {}, item num of dict: {}".format(vocab_size, len(wordId_wordEmbed_dict.keys()))
        
        for param in self.ye.params():
            for i in range(vocab_size):
                if i in wordId_wordEmbed_dict:
                    param.data[i,:] = wordId_wordEmbed_dict[i]
            break
        print "word imbed initialized"
        """

    def __call__(self,y,cell,h,context,train=True):
        e = F.tanh(self.ye(y))
        #cell_new,h_new = F.lstm(cell,F.dropout(self.eh(e)+self.hh(h)+self.ch(context),ratio=self.dropout_lstm_ratio,train=train))
        cell_new,h_new = F.lstm(cell,self.eh(e)+self.hh(h)+self.ch(context))
        h_new = F.dropout(h_new,ratio=self.dropout_ratio,train=train)
        f = F.tanh(self.hf(h_new)+self.cf(context)+e)
        #should add a tanh after L_0.
        return self.L0(F.dropout(f,ratio=self.dropout_ratio,train=train)),cell_new,h_new


def SoftAtt(e, context,train=True):
    #e (bs,14*14)
    #context (bs,14*14,512)
    bs = e.data.shape[0]
    return F.batch_matmul(F.reshape(e,(bs,1,14*14)),context)
    
def HardAtt(e, context,train=True):
    #multinomial is not support by cupy , so convert 
    #1) gpu -> cpu, 
    #2) use numpy.random.multinomial
    #3) cpu -> gpu
    bs = e.data.shape[0]
    e_list_np = cuda.to_cpu(e.data) #(bs,14*14)
    context_list_np = cuda.to_cpu(context.data) #(bs,14*14,512)    
    context = np.zeros((bs,1,512)).astype(np.float32)
    e_idx = np.zeros((bs,14*14))
    for i in range(bs):
        prob = e_list_np[i]/np.sum(e_list_np[i])
        prob = prob - np.finfo(np.float32).epsneg  
        try:
            #idx = np.argmax(prob)
            idx = np.random.multinomial(1,prob).argmax()  
        except:
            print "prob sum {}".format(np.sum(prob))
            idx = np.argmax(prob)
            #print "idx is {}, prob is {}".format(idx,prob[idx])
        

        #idx = np.argmax(prob)
        e_idx[i][idx] = 1.0
        context[i][0] = context_list_np[i][idx]
    ctx_v = Variable(cuda.to_gpu(context),volatile=not train)
    return ctx_v,e_idx.astype(np.float32) 
    

class CaptionGenNet(chainer.Chain):
    def __init__(self,vocab_size, embed_size,context_size,hidden_size,n_layer_init,
                 n_layer_att,dropout_ratio,wordId_wordEmbed_dict_file,coeff1,coeff2,coeff3):
        super(CaptionGenNet,self).__init__(
            init_c=MultiNet_init(context_size,hidden_size,n_layer_init,dropout_ratio),
            init_h=MultiNet_init(context_size,hidden_size,n_layer_init,dropout_ratio),
            att=Attention(hidden_size,context_size,n_layer_att),
            bg=L.Linear(hidden_size,1),
            dec=Decoder(vocab_size,embed_size,context_size,hidden_size,dropout_ratio,wordId_wordEmbed_dict_file)
            )

        self.n_layer_att = n_layer_att
        self.n_layer_init = n_layer_init
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.coeff_att = coeff1
        self.coeff_entropy = coeff2
        self.coeff_reinforce = coeff3
        self.semi_sampling_p = 0.5
        self.baseline = 0
        print "load successfully"

    def encode(self,context,train=None):
        #context array-->(batchsize, 14*14,512)
        bs = context.data.shape[0]
        self.context = context 
        #need activation function : see build_sampler() line 748
        self.lstm_cell = self.init_c(F.sum(context,axis=1)/(14*14),train=train)
        self.lstm_h = self.init_h(F.sum(context,axis=1)/(14*14),train=train)

    def decode(self,prev_y,t=None,train=True):
        e = self.att(self.context,self.lstm_h,train=train) #(bs,14*14)
        bs = e.data.shape[0]
        context = SoftAtt(e,self.context,train=train) #(bs,1,512)
        beta_gate = F.sigmoid(self.bg(self.lstm_h))
        contextBeta = F.reshape(F.batch_matmul(beta_gate,context),(bs,512))
        y, self.lstm_cell, self.lstm_h = self.dec(prev_y,self.lstm_cell,self.lstm_h,contextBeta,train)
        if t is not None:
            loss = F.softmax_cross_entropy(y,t)
            return loss,e
        else:
            return y

    def decode_hard(self,prev_y, t=None,train=True):
        #if t is None:
        #    print "hard"
        e = self.att(self.context,self.lstm_h)
        if t is not None:
            h_sampling_mask = cuda.to_gpu(np.random.binomial(1,p=self.semi_sampling_p))
            bs = e.data.shape[0]
            context = h_sampling_mask*SoftAtt(e,self.context,train=train)
            hard_c, e_idx =  HardAtt(e,self.context,train=train)
            context = context + (1-h_sampling_mask)*hard_c
        else:
            # e: ( bs, 14*14)
            #self.context (bs,14*14,512)
            #print "hard hard"
            idx = np.argmax(cuda.to_cpu(e.data),axis=1)
            context = F.split_axis(self.context,indices_or_sections=14*14,axis=1)[idx]
        #beta_gate = F.sigmoid(self.bg(self.lstm_h))
        #contextBeta = F.reshape(F.batch_matmul(beta_gate,context),(bs,512))
        y, self.lstm_cell, self.lstm_h = self.dec(prev_y,self.lstm_cell,self.lstm_h,context,train)
        if t is not None:
            loss = F.softmax_cross_entropy(y,t) 
            return loss, e, e_idx
        else:
            return y

    def get_max_k(self,x, prob, k=3):
        #sorted_xi = sorted(enumerate(x), key=lambda _:_[1],reverse=True)
        #return sorted_xi[:k]
        try:
            mask = np.random.choice(len(x),k,p=prob,replace=False)
        except:
            #print "sum prob : {}".format(np.sum(prob))
            sorted_xi = sorted(enumerate(x), key=lambda _:_[1],reverse=True)
            mask = [i[0] for i in sorted_xi]
        return zip(mask,[x[i] for i in mask])

    def gen_sample(self,context,end=0,width=3,maxlen=15,soft="Soft"):
        #context shape : (bs,512,14*14)
        self.encode(context)
        cands = Queue.PriorityQueue() # element: (score, sequence, current state)
        cands.put((0.0,[2],[self.lstm_cell,self.lstm_h]))

        count = 0
        while True:
            count += 1
            next_queue = Queue.PriorityQueue()
            updated = False
            while not cands.empty():
                score, seq, state = cands.get()
                if (len(seq) > 1 and seq[-1] == end) or len(seq) > maxlen:
                    next_queue.put((score,seq,state))
                    continue
                self.lstm_cell = state[-2]
                self.lstm_h = state[-1]
                x = Variable(cuda.to_gpu(np.array([seq[-1]],dtype=np.int32)),volatile=True)
                if soft == "Soft":
                    y = F.softmax((self.decode(x,train=False)))
                else:
                    y = F.softmax((self.decode_hard(x,train=False)))
                new_state = [self.lstm_cell, self.lstm_h]
                prob = cuda.to_cpu(y.data).reshape(-1,)
                prob = prob/prob.sum()
                prob = prob - np.finfo(np.float32).epsneg 
                scores = cuda.to_cpu(F.log(y).data).reshape(-1,)
                for i,y_ in self.get_max_k(scores,prob,k=width):
                    next_queue.put((score+y_,seq+[i],state+[self.lstm_cell,self.lstm_h]))
                    updated = True
                while next_queue.qsize() > width:
                    next_queue.get(False)
            cands = next_queue
            if not updated:
                break
        while cands.qsize() >= 2:
            cands.get(False)
        assert cands.qsize() == 1
        ll,seq, states = cands.get(False)
        return ll,seq,states


    def compute_graph(self,xs,ts,words_len,bs,kind="Hard",train=True,cross_entropy_loss=False,dir_name="."):
        if kind == "Soft":
            loss = 0
            e = []
            for x, t in zip(xs,ts):
                tmp_loss,e_t = self.decode(x,t,train=True) #eVar : (bs,14*14)

                import chainer.computational_graph as c
                print "type is {}".format(type(tmp_loss))
                g = c.build_computational_graph((tmp_loss,))
                with open("{}/arctic_caption_one_step.dot".format(dir_name),'w') as o: 
                    o.write(g.dump())         
                break



    def soft_loss(self,xs,ts,words_len,bs,train=True,cross_entropy_loss=False):
        loss = 0
        e = []
        for x,t in zip(xs,ts):
            tmp_loss,e_t = self.decode(x,t,train=train) #eVar : (bs,14*14)
            e.append(e_t)
            loss += tmp_loss

        eVariable = F.transpose(F.stack(e),(1,0,2)) # (words_len,bs,length)
        words_len = len(xs)*1.0
        e_loss = F.sum((words_len/196-F.sum(eVariable,axis=1))**2)

        #eVar_list = F.split_axis(eVariable,indices_or_sections=bs,axis=0) #(bs,words_len,length)
        #e_loss = 0
        #for i in range(bs):
        #   tmp_e = F.sum(eVar_list[i],axis=0)
        #   e_loss += F.sum((words_len/196-tmp_e)*(words_len/196-tmp_e))

        e_loss = e_loss*self.coeff_att/bs
        final_loss = loss + e_loss
        if cross_entropy_loss is True:
            return loss
        else:
            return final_loss, e_loss

    def hard_loss(self,xs,ts,words_len,bs,train=True,cross_entropy_loss=False):
        loss  = 0
        entropy_alpha = 0
        reinforce_e = 0
        e = []
        e_idx_list = []
        reward = 0
        #Entropy -sum(p(s)*log p(s))
        for x, t in zip(xs,ts):
            tmp_loss, tmp_e, e_idx = self.decode_hard(x,t,train=train)
            #tmp_loss : (1,), tmp_e: (bs,c_length), e_idx: (bs,1)
            loss += tmp_loss
            reward = reward - tmp_loss.data
            e.append(tmp_e)
            e_idx_list.append(e_idx)
        eVariable = F.transpose(F.stack(e),(1,0,2)) #(words_len,bs,length)

        #For entropy loss
        eps = np.zeros(eVariable.data.shape).astype(np.float32)
        eps[:] = np.finfo(np.float32).epsneg
        epsVar = Variable(cuda.to_gpu(eps),volatile=not train)
        e_alpha = F.where(eVariable.data > 0, eVariable,epsVar)
        entropy_alpha = F.sum(e_alpha*F.log(e_alpha))/(-1*bs)
        
        #For reinforce loss
        #To-do
        
    def __call__(self,xs,ts,words_len,bs,kind="Hard",train=True,cross_entropy_loss=False):
        if kind == "Soft":
            return self.soft_loss(xs,ts,words_len,bs,train=train,cross_entropy_loss=cross_entropy_loss)
        else:
            return self.hard_loss(xs,ts,words_len,bs,train=train,cross_entropy_loss=cross_entropy_loss)

