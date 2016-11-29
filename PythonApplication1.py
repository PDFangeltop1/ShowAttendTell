import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

import skimage
from skimage import io
from skimage.transform import resize

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.functions import caffe
from chainer import cuda
from chainer import optimizers,serializers,Variable
import numpy as np
import cPickle as cp
import argparse
import os.path
import net
import json
import utils
import homo_data
import solver

from bleu import Bleu

def cap_generate():
    _dir = "../WholeModel"
    utils.write_hyper_to_file()
    hyper_param_setting = utils.load_data_from_disk("{}/model_hyper_setting".format(_dir))
    parser = argparse.ArgumentParser(description="attention classify program")
    parser.add_argument('--e',dest='n_epoch',default=hyper_param_setting["e"],type=int)
    parser.add_argument('--n1',dest='n_layer_init',default=hyper_param_setting["n1"],type=int)
    parser.add_argument('--n2',dest='n_layer_att',default=hyper_param_setting["n2"],type=int)
    parser.add_argument('--b',dest='batchsize',default=hyper_param_setting["b"],type=int)
    parser.add_argument('--hh',dest='n_h',default=hyper_param_setting["h"],type=int)
    parser.add_argument('--nw',dest='n_word',default=hyper_param_setting["nw"],type=int)
    parser.add_argument('--gpu',dest='use_gpu',default=hyper_param_setting["gpu"],type=int)
    parser.add_argument('--lr',dest='learning_rate',default=hyper_param_setting["lr"],type=float)
    parser.add_argument('--d',dest='dropout_ratio',default=hyper_param_setting["d"],type=float)  
    parser.add_argument('--c1',dest='coeff_att',default=hyper_param_setting["c1"],type=float) 
    parser.add_argument('--c2',dest='coeff_entropy',default=hyper_param_setting["c2"],type=float)
    parser.add_argument('--c3',dest='coeff_reinforce',default=hyper_param_setting["c3"],type=float)
    parser.add_argument('--v',dest='vocab_size',default=hyper_param_setting["v"],type=int)  
    parser.add_argument('--estop',dest='early_stop_freq',default=hyper_param_setting["estop"],type=int)
    parser.add_argument('--trainVal',dest='train_val_number',default=hyper_param_setting["trainVal"],type=int)
    parser.add_argument('--p',dest='patience',default=hyper_param_setting["p"],type=int)
    args = parser.parse_args()
 
    #Training Data: [[[word_id_1,word_id_2,....],feature_id],......] 
    date = "20161122"
    epoch_str = "Soft_latest_epoch"
    #epoch_str = "Hard_epoch_10"
    load_model = ["{}/{}/{}.model".format(_dir,date,epoch_str),"{}/{}/{}.state".format(_dir,date,epoch_str)]
    data = homo_data.HomogeneousData(hyper_param_setting,indexKey="train_data",batch_size=args.batchsize,maxlen=50)
    val_data = homo_data.HomogeneousData(hyper_param_setting,indexKey="val_data",batch_size=args.batchsize,maxlen=50)
    model = net.CaptionGenNet(args.vocab_size,args.n_word,512,args.n_h,args.n_layer_init,args.n_layer_att,args.dropout_ratio,
                              hyper_param_setting,args.coeff_att,args.coeff_entropy,args.coeff_reinforce)
    mysolver = solver.SolverTqdm(model,data,val_data,hyper_param_setting,args.batchsize,args.n_epoch,args.learning_rate,args.early_stop_freq,
                                 args.train_val_number,args.patience,args.use_gpu,load_model=None,load_db=True,soft="Soft",opt="Adam")
    date = "20161122_tensor_param_train"
    filename = "{}/{}".format(_dir,date)
    try:
        os.mkdir(filename)
    except:
        print "directory already existed"
    mysolver.train(filename,hyper_param_setting["jsonFile"])
    #mysolver.compute_graph(filename)
    #mysolver.visualize_attention(filename,train=False)



"""
the reason why soft goes well:
1 in the alpha computation part:   should not use split alpha list along batch axis.
eVariable = F.transpose(F.stack(e),(1,0,2))
words_len = len(xs)*1.0
e_loss = F.sum((words_len/196-F.sum(eVariable,axis=1))**2) 

not this:
eVar_list = F.split_axis(eVariable,indices_or_sections=bs,axis=0) #(bs,words_len,length)
for i in range(bs): 
   tmp_e = F.sum(eVar_list[i],axis=0) 
   e_loss += F.sum((words_len/196-tmp_e)*(words_len/196-tmp_e))  

2 not Sum((1-sum(alpha))), but Sum(length_of_sents-sum(alpha))

3 activation function:
  Do not use tanh in attention computation part , use relu instead!


How to let hard go well too?
"""
if __name__ == "__main__":
    #utils.write_hyper_to_file()
    cap_generate()
    #test_vgg()
    #utils.get_word_id_dict("../TinyModel")
    #utils.get_wordId_wordEmbed_dict("../TinyModel")
    #utils.save_imgFeature_imgFeatureId("../WholeModel")
    #utils.save_img_feature_to_file("../WholeModel")
    #utils.save_npy_to_db("../TinyModel")
