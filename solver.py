import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import skimage
from skimage import io
from skimage.transform import resize
import chainer
import sys
import chainer.functions as F
import chainer.links as L
from chainer import optimizers,serializers,Variable
from chainer import cuda
import numpy as np
import utils
from tqdm import tqdm
from bleu import Bleu
import json

class SolverTqdm(object):
    def __init__(self,model,data,val_data,index_files,batchsize,n_epoch,learning_rate,
                 early_stop_freq,train_val_number,patience,use_gpu,load_model=None,load_db=True,soft="Soft",opt="Adam"):
        self.data = data
        self.val_data = val_data
        self.batchsize = batchsize
        self.n_epoch = n_epoch
        self.use_gpu = (use_gpu>=0)
        self.early_stop_freq = early_stop_freq
        self.train_val_number = train_val_number
        self.patience = patience  #early stop
        self.load_db = load_db
        if opt == "Adam":
            self.optimizer = optimizers.Adam(alpha=learning_rate)  #for MSCOCO, use adam, otherwise it will get nan. 
            print "use Adam "
        else:
            self.optimizer = optimizers.NesterovAG(lr=learning_rate)
            print "use NesterovAG"
        self.model = model
        self.learning_rate = learning_rate
        self.soft = soft
        self.model_hyper_str = "optimizer_{}_attentionKind_{}_batchsize_{}_nepoch_{}_hiddenSize_{}_learnrate_{}".format(opt,self.soft,batchsize,n_epoch,
                                                                                                           self.model.hidden_size,learning_rate)
        self.index_files = index_files
        d = utils.load_data_from_disk(index_files["word_id"])
        self.id_word_dict = {v:k for k,v in d.iteritems()}
        self.id_word_dict[0] = "<eos>"
        self.id_word_dict[1] = "UNK"
        self.id_word_dict[2] = "<bos>"

        if load_model is not None:
            #serializers.load_hdf5("cfy_ram_hdf5_0830\ram_epoch200.chainermodel",self.model)
            serializers.load_npz(load_model[0],self.model)
            self.optimizer.setup(self.model)
            serializers.load_npz(load_model[1],self.optimizer)
            #self.optimizer.setup(self.model)
            #print self.optimizer.lr
            print "load model successfully!"

        if self.use_gpu:
            chainer.cuda.get_device(use_gpu).use()
            self.model.to_gpu()
            self.model.use_gpu = True
        self._reset()
    
    def _reset(self):
        self.best_val_loss = 0
        self.loss_history = []
        self.val_likelihood_history = []

    def visualize_attention(self,_dir,train=True):
        if self.load_db is False:
            self.val_imgFeatureIDimgFeature_dict = utils.load_data_from_disk(self.index_files["val_id_feat"])
            print "load val image feature successfully, data size {}".format(len(self.val_imgFeatureIDimgFeature_dict.keys()))

        imgID_FeatureID_dict = utils.load_data_from_disk(self.index_files["img_id"])
        featId_imgId = {v:k for k, v in imgID_FeatureID_dict.iteritems()}
        json_file = self.index_files["jsonFile"]
        data_dir = "/home/angeltop1/DLResearch/Google_Refexp_toolbox/external/coco"
        COCOdata = json.load(open(json_file,'r'))
        imgId_filename = {}
        for i in range(len(COCOdata["images"])):
            imgId_filename[COCOdata["images"][i]["id"]] = COCOdata["images"][i]["file_name"]
        
        #print "visual atttention!"
        prefix = "val2014"
        count = 0
        for i in range(len(COCOdata["images"])):
            if count > 40:
                break
                
            if COCOdata["images"][i]["id"] not in imgID_FeatureID_dict:
                continue
        
            idx = imgID_FeatureID_dict[COCOdata["images"][i]["id"]]
            #if idx not in self.val_imgFeatureIDimgFeature_dict:
            #    continue

            count += 1
            if self.load_db is False:
                c = self.val_imgFeatureIDimgFeature_dict[idx]
                c = np.transpose(c,(1,0)).reshape(1,14*14,512).astype(np.float32)
            else:
                c = utils.load_data_from_db(self.index_files["val_id_feat"],idx)
                c = np.transpose(c,(0,2,1)).astype(np.float32)

            #print "read context !!"
            context = Variable(cuda.to_gpu(c),volatile=True)
            _, gen_sample, states = self.model.gen_sample(context)

            #print "generate sents!!"
            I = io.imread("{}/images/{}/{}".format(data_dir,prefix,COCOdata["images"][i]["file_name"]))
            I_crop = utils.crop_img(I)

            H = int(np.sqrt(len(gen_sample)+1))
            if H*H < len(gen_sample) + 1:
                H += 1

            #print "####JFIDJFLDJLFDOIFIJDFLJDLIFJILDFbefore subplots"
            #print "H : {}".format(H)
            #plt.ioff()
            #print "type of plt: {}".format(type(plt))
            fig,ax = plt.subplots(H,H)
            #print "before imshow !"
            ax.flatten()[0].imshow(I_crop)
            ax.flatten()[0].get_xaxis().set_visible(False)
            ax.flatten()[0].get_yaxis().set_visible(False)
            plt.set_cmap(matplotlib.cm.Greys_r)          

            #print "after imshow!!"
            gen_sents = []
            for ai, (i, id) in zip(ax.flatten()[1:],enumerate(gen_sample)):
                ai.imshow(I_crop)
                ai.get_xaxis().set_visible(False)
                ai.get_yaxis().set_visible(False)
                if i >= len(gen_sample) - 2:
                    continue
                cur_word = self.id_word_dict[gen_sample[i+1]]
                gen_sents.append(cur_word)
                ai.text(0,1,cur_word)
                alphaVar = self.model.att(context,states[2*i+1])
                alpha = cuda.to_cpu(alphaVar.data)
                #print "alpha shape: {}".format(alpha.shape)

                #print "cuda to cpu! {}".format(i)
                alpha = alpha.reshape(14,14)
                alpha_img = skimage.transform.pyramid_expand(alpha,upscale=16,sigma=20)
                print "alpha min: {}, max : {}".format(np.min(alpha),np.max(alpha))
                #alpha_img = skimage.transform.resize(alpha,[I_crop.shape[0],I_crop.shape[1]])
                ai.imshow(alpha_img,alpha=0.8)
            
            for ai in ax.flatten():
                ai.get_xaxis().set_visible(False)
                ai.get_yaxis().set_visible(False)

            print "alpha shape: {}".format(alpha.shape)
            print " ".join(gen_sents)
            import os
            try:
                os.mkdir("{}/figs".format(_dir))
            except:
                print "directory already existed"

            if train is True:
                fig.savefig("{}/figs/epoch_{}_fig_{}.png".format(_dir,self.epoch,count))
            else:
                fig.savefig("{}/figs/fig_{}.png".format(_dir,count))                
    def train_val(self,file_name, jsonFile="C:/Users/Ran Wensheng/Desktop/CaptionGenModelData/MSCOCO_data/annotations/captions_val2014.json"):
        json_file = jsonFile
        COCOdata = json.load(open(json_file,'r'))

        imgId_annotationId = {}
        for i in range(len(COCOdata["annotations"])):
            if COCOdata["annotations"][i]["image_id"] in imgId_annotationId:
                pass
            else:
                imgId_annotationId[COCOdata["annotations"][i]["image_id"]] = []
            imgId_annotationId[COCOdata["annotations"][i]["image_id"]].append(i)

        count = 0
        total_blue_score = {"Bleu_4":0,"Bleu_3":0,"Bleu_2":0,"Bleu_1":0}
        #for batch_idx in self.data:
        for i in range(len(COCOdata["images"])):
            if COCOdata["images"][i]["id"] not in self.imgIdFeatureId:
                continue
            idx = self.imgIdFeatureId[COCOdata["images"][i]["id"]]
            if self.load_db is False:
                c = self.val_imgFeatureIDimgFeature_dict[idx]
                c = np.transpose(c,(1,0)).reshape(1,14*14,512).astype(np.float32)
            else:
                c = utils.load_data_from_db(self.index_files["val_id_feat"],idx)
                c = np.transpose(c,(0,2,1)).astype(np.float32)
            context = Variable(cuda.to_gpu(c),volatile=True)
            _, gen_sample, _ = self.model.gen_sample(context,soft=self.soft)
            gen_sample = gen_sample[1:]
            gen_sents = " ".join([self.id_word_dict[id] for id in gen_sample])

            ref_sents = []
            for annoId in imgId_annotationId[COCOdata["images"][i]["id"]]:
                ref_sents.append(COCOdata["annotations"][annoId]["caption"])

            if count%500 == 0:
                self.write_to_file("generate sents : {}".format(gen_sents),dir_name=file_name,file_name="generate_sents_file")
                self.write_to_file("reference sents : {}".format(ref_sents[0]),dir_name=file_name,file_name="generate_sents_file")

            #how to calculate bleu score?
            tmp_blue_score = utils.score({1:ref_sents},{1:[gen_sents]})
            for k,v in tmp_blue_score.iteritems():
                total_blue_score[k] +=v

            count += 1
            #print count
            if count >= self.train_val_number:
                break
        for k,v in total_blue_score.iteritems():
            total_blue_score[k] /= count
        print total_blue_score
        return total_blue_score

    def idx_to_array(self,batch_idx,train=True):
        #Training Data: [[[word_id_1,word_id_2,....],feature_id],......] 
        batchsize = len(batch_idx)
        if train is True:
            words_len = len(self.data.data[batch_idx[0]][0]) 
        else:
            words_len = len(self.val_data.data[batch_idx[0]][0]) 
        xs = []
        ts = []
        c = np.zeros((batchsize,512,14*14))
        for j in range(words_len-1):
            x = np.zeros((batchsize,))
            t = np.zeros((batchsize,))
            for i,bs in enumerate(batch_idx):
                if train is True:
                    x[i] = self.data.data[bs][0][j]
                    t[i] = self.data.data[bs][0][j+1]
                else:
                    x[i] = self.val_data.data[bs][0][j]
                    t[i] = self.val_data.data[bs][0][j+1]
            xs.append(Variable(cuda.to_gpu(x.astype(np.int32)),volatile=not train))
            ts.append(Variable(cuda.to_gpu(t.astype(np.int32)),volatile=not train))

        if self.load_db is False:
            for i,bs in enumerate(batch_idx):
                if train is True:
                    idx = self.data.data[bs][1]
                    c[i,:,:] =  self.train_imgFeatureIDimgFeature_dict[idx]
                    
                else:
                    idx = self.val_data.data[bs][1]
                    c[i,:,:] =  self.val_imgFeatureIDimgFeature_dict[idx]
        else:
            if train is True:
                ids = [self.data.data[bs][1] for bs in batch_idx]
                c = utils.load_data_from_db(self.index_files["train_id_feat"],ids=ids)
            else:
                ids = [self.val_data.data[bs][1] for bs in batch_idx]
                c = utils.load_data_from_db(self.index_files["val_id_feat"],ids=ids)
                
                
        #return context shape (bs,14*14,512)
        c = Variable(cuda.to_gpu(c.astype(np.float32)),volatile=not train)
        return xs,ts,F.transpose(c,(0,2,1)), words_len,batchsize

    def get_val_likelihood(self):
        self.val_data.reset()
        total_neg_likelihood = 0
        count = 0
        for batch_idx in self.val_data:
            xs,ts,context, words_len,bs = self.idx_to_array(batch_idx,train=False)
            #context -->(batchsize, 512,14*14)
            self.model.encode(context)
            loss = self.model(xs,ts,words_len,bs,kind=self.soft,train=False,cross_entropy_loss=True)
            total_neg_likelihood += cuda.to_cpu(loss.data)
            count += 1
            
            """
            import chainer.computational_graph as c
            g = c.build_computational_graph((loss,))
            with open("arctic_caption.dot",'w') as o:
                o.write(g.dump())
            """

            if count > self.early_stop_freq//20:
                break

        return total_neg_likelihood/count
        
    def compute_graph(self,dir_name):
        self.val_data.reset()
        total_neg_likelihood = 0
        count = 0
        for batch_idx in self.val_data:
            xs,ts,context, words_len,bs = self.idx_to_array(batch_idx,train=True)
            #context -->(batchsize, 512,14*14)
            self.model.encode(context,train=True)
            self.model.compute_graph(xs,ts,words_len,bs,kind=self.soft,train=True,cross_entropy_loss=True,dir_name=dir_name)
            break
        
        
    def _step(self,batch_idx):
        xs,ts,context, words_len,bs = self.idx_to_array(batch_idx)
        #context -->(batchsize, 512,14*14)
        self.model.encode(context,train=True)
        loss, e_loss = self.model(xs,ts,words_len,bs,kind=self.soft,train=True)

        
        self.model.zerograds()
        loss.backward()
        loss.unchain_backward()
        #print "Before update,grad normal is {}".format(self.optimizer.compute_grads_norm())
        self.optimizer.update()
        #print "After update,grad normal is {}".format(self.optimizer.compute_grads_norm())
        if self.soft == "Soft":
            return cuda.to_cpu(loss.data) , cuda.to_cpu(e_loss.data)
        else:
            if type(e_loss[0]) == int:
                e0 = e_loss[0]
            else:
                e0 = cuda.to_cpu(e_loss[0].data)

            if type(e_loss[1]) == int:
                e1 = e_loss[1]
            else:
                e1 = cuda.to_cpu(e_loss[1].data)

            return cuda.to_cpu(loss.data) , [e0,e1]

    def write_to_file(self,_str,dir_name,file_name="classify_train_val_loss_array"):
        print _str
        f = open("{}/{}".format(dir_name,file_name),'a')
        f.write(_str)
        f.write("\n")
        f.close()
    
    def train(self,file_name,
              jsonFile="C:/Users/Ran Wensheng/Desktop/CaptionGenModelData/MSCOCO_data/annotations/captions_val2014.json"):
        self.imgIdFeatureId = utils.load_data_from_disk(self.index_files["img_id"])
        if self.load_db is False:
            self.train_imgFeatureIDimgFeature_dict = utils.load_data_from_disk(self.index_files["train_id_feat"])
            print "load train image feature successfully, data size {}".format(len(self.train_imgFeatureIDimgFeature_dict.keys()))
            self.val_imgFeatureIDimgFeature_dict = utils.load_data_from_disk(self.index_files["val_id_feat"])
            print "load val image feature successfully, data size {}".format(len(self.val_imgFeatureIDimgFeature_dict.keys()))
        
            
        self.optimizer.setup(self.model)
        self.optimizer.add_hook(chainer.optimizer.GradientClipping(5))
        num_iterations = 0
        lr_gamma = np.exp(-3*np.log(10)/self.n_epoch)
        self.write_to_file(self.model_hyper_str,file_name)

        bad_counter = 0 #for early stop
        estop = False 
        for epoch in range(self.n_epoch):
            self.epoch = epoch
            #self.optimizer.lr = self.learning_rate * np.power(lr_gamma, self.epoch)
            #self.write_to_file("learning rate : {}".format(self.optimizer.lr),file_name)
            self.data.reset()
            for batch_idx in self.data:
                num_iterations += 1
                loss,e_loss = self._step(batch_idx)                    
                if num_iterations%200 == 0:
                    if self.soft == "Hard":
                        _str = "Epoch {} , Iteration {}:,  loss: {}, reinforce_loss: {},entropy loss: {}".format(self.epoch,num_iterations,loss,e_loss[0], e_loss[1])
                        baseline_str = "Epoch {}, Iteration {} : baseline {}".format(self.epoch,num_iterations,self.model.baseline)
                        self.write_to_file(baseline_str,dir_name=file_name,file_name="baseline_histroy_log_file")
                    else:
                        _str = "Epoch {} , Iteration {}:,  loss: {}, e_loss: {}".format(self.epoch,num_iterations,loss,e_loss)
                    self.write_to_file(_str,dir_name=file_name,file_name="loss_history_file")


                #early stop!
                #if num_iterations%self.early_stop_freq == 0:
                if num_iterations%self.early_stop_freq == 0:
                    neg_likelihood = self.get_val_likelihood()
                    if num_iterations%(self.early_stop_freq*10) == 0:
                        print "neg_likelihood {}".format(neg_likelihood)
                    self.val_likelihood_history.append(neg_likelihood)
                    if neg_likelihood <= np.array(self.val_likelihood_history).min():
                        bad_counter = 0

                    if epoch > self.patience and len(self.val_likelihood_history) > self.patience and neg_likelihood >= np.array(self.val_likelihood_history)[:-self.patience].min():
                        bad_counter += 1
                        if bad_counter > self.patience:
                            print "early stop!"
                            estop = True
                            break

            if self.epoch%5 == 0:
                serializers.save_npz("{}/{}_epoch_{}.model".format(file_name,self.soft,self.epoch),self.model)
                serializers.save_npz("{}/{}_epoch_{}.state".format(file_name,self.soft,self.epoch),self.optimizer)

            serializers.save_npz("{}/{}_latest_epoch.model".format(file_name,self.soft),self.model)
            serializers.save_npz("{}/{}_latest_epoch.state".format(file_name,self.soft),self.optimizer)
            self.visualize_attention(_dir=file_name)
            if self.epoch < self.patience:
                continue

            val_acc = self.train_val(file_name,jsonFile)
            self.loss_history.append(loss)

            log_str ="(Epoch %d/%d) train loss: %f, val bleu1: %f, bleu2: %f, bleu3: %f, bleu4: %f"%(
                self.epoch,self.n_epoch,loss,val_acc["Bleu_1"],val_acc["Bleu_2"],val_acc["Bleu_3"],val_acc["Bleu_4"])
            self.write_to_file(log_str,file_name)

            if estop:
                break
 
