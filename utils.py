import sys
import json
import re
import numpy as np
import skimage
from skimage import io
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.functions import caffe
from chainer import cuda
from chainer import optimizers,serializers,Variable
from skimage.transform import resize
from bleu import Bleu

regex = re.compile('[\:\(\);",\.!?]')


import lmdb
def load_data_from_db(lmdb_name,ids=None):
    ids = ids if type(ids) == list else [ids]
    N = len(ids)
    feats = np.zeros((N,512,196))
    env = lmdb.open(lmdb_name,readonly=True)
    with env.begin() as txn:
        for i,idx in enumerate(ids):
            str_id = str(idx)
            feat = np.fromstring(txn.get(str_id),dtype=np.float32).reshape(512,196)
            feats[i,:,:] = feat
    return feats

def save_data_to_db(lmdb_name, features, ids):
    LMDB_MAP_SIZE = 1099511627776
    env = lmdb.open(lmdb_name,map_size=LMDB_MAP_SIZE)
    N = len(features)
    with env.begin(write=True) as txn:
        for i in range(N):
            str_id = str(ids[i])
            txn.put(str_id,features[i].reshape(-1,))
            if i% 1000 == 0:
                print i
    
def save_dict_to_disk(d,filename):
    import cPickle as cp
    with open(filename,'wb') as f:
        cp.dump(d,f,protocol=cp.HIGHEST_PROTOCOL)

def load_data_from_disk(filename):
    import cPickle as cp
    with open(filename,'rb') as f:
        data = cp.load(f)
    return data

def get_word_id_dict(dirname):
    json_files = [#"/home/angeltop1/DLResearch/Google_Refexp_toolbox/external/coco/annotations/captions_train2014.json",
                  "/home/angeltop1/DLResearch/Google_Refexp_toolbox/external/coco/annotations/captions_val2014.json"]
    word_id_dict = {}
    id_count = 3 #<eos> 0 , UNK 1, <bos> 2,
    for json_file in json_files:
        print "processing file : {}".format(json_file)
        data = json.load(open(json_file,'r'))
        print data.keys()
        print "annotation type : {}, length: {}, keys: {}".format(type(data["annotations"]),len(data["annotations"]),data["annotations"][0].keys())
        print "images type : {}, length: {}, keys; {}".format(type(data["images"]),len(data["images"]),data["images"][0].keys())

        length = len(data["annotations"])//20  #//20 For TinyModel
        for i in range(length):
            sents = data["annotations"][i]["caption"].strip().split()
            new_sents = []
            length = len(sents)
            for ii, word in enumerate(sents):
                new_sents.append(regex.sub('',word))
                if re.search(regex,word):
                    if not word[0].isalpha():
                        new_sents.append(word[0]) 
                    if not word[-1].isalpha():    
                        new_sents.append(word[-1])
                else:
                    if ii == length - 1:
                        new_sents.append(".")
            if i% 30000 == 0:
                print " ".join(new_sents)
            for word in new_sents:
                if word in word_id_dict:
                    word_id_dict[word] += 1
                else:
                    word_id_dict[word] = 1
                    id_count += 1

    print "dict size is {}".format(len(word_id_dict.keys()))
    wordcount_1 = 1
    count_1 = 0
    w1c = 0

    wordcount_2 = 2
    count_2 = 0
    w2c = 0

    wordcount_3 = 3
    count_3 = 0
    w3c = 0

    all_word_count = 0
    word_in_d1 = 0
    w5c = 0
    w6c = 0
    w7c = 0
    for k,v in word_id_dict.iteritems():
        all_word_count += v
        if v == 1:
            w1c += 1

        if v == 2:
            w2c += 2

        if v == 3:
            w3c += 3

        if v > wordcount_1:
            count_1 += 1
        if v > wordcount_2:
            count_2 += 1
        if v > wordcount_3:
            count_3 += 1

        if v > 4:
            word_in_d1 += 1
        
        if v > 5:
           w5c += 1 
           
        if v > 6:
            w6c += 1
            
        if v > 7:
            w7c += 1

    print "word appears 1 times : {}, 2 times: {}, 3 times : {}".format(count_1,count_2,count_3)
    print "all word : {}, word appears 1 times : {}, 2 times: {}, 3 times : {}".format(all_word_count,w1c,w2c,w3c)
    print "percent 1: {}, 2: {} ,3 :{}".format(w1c*1.0/all_word_count,w2c*1.0/all_word_count,w3c*1.0/all_word_count)
    id_count = 3  #<eos> 0 , UNK 1, <bos> 2
    word_id_dict_1 = {}
    for k,v in word_id_dict.iteritems():
        if v > 6:
            word_id_dict_1[k] = id_count
            id_count += 1

    #For train and val dataset,
    #word appears 1 times : 21419, 2 times: 16650, 3 times : 14152
    #12533 words appears more than 4 times, 11339 words more than 5 times,10503 words 6 times, 9808 words 7 times
    print "{} words appears more than 4 times, {} words more than 5 times,{} words 6 times, {} words 7 times,{} words in word vocabulary".format(word_in_d1,w5c,w6c,w7c,len(word_id_dict_1.keys()))

    #save_dict_to_disk(word_id_dict_1,"{}/word_id_dict_6More_train_val_2014.npy".format(dirname)) #10503
    save_dict_to_disk(word_id_dict_1,"{}/word_id_dict_6More_small_val_2014.npy".format(dirname)) #1119
    return word_id_dict_1

def get_wordId_wordEmbed_dict(dirname):
    #wordVectorFile = "/home/angeltop1/GloveVector/glove.840B.300d.txt"
    wordVectorFile = "/home/angeltop1/word2vec/vectors.bin"
    f = open(wordVectorFile,'r')

    #word_id_dict = load_data_from_disk("{}/word_id_dict_6More_train_val_2014.npy".format(dirname)) 
    word_id_dict = load_data_from_disk("{}/word_id_dict_6More_small_val_2014.npy".format(dirname))  #TinyModel val//20
    line = f.readline()
    count = 0
    wordId_wordEmbed_dict = {}
    while line:
        count += 1
        if count % 100000 == 0:
            print len(wordId_wordEmbed_dict.keys())
        array = line.strip().split(' ')
        word = array[0]
        embed = array[1:]
        #print "embed type : {}, value : {}".format(type(array[1]),array[1])
        if word in word_id_dict:
            #if word_id_dict[word] not in wordId_wordEmbed_dict:
            #    print word
            wordId_wordEmbed_dict[word_id_dict[word]] = embed 
        line = f.readline()
 
        
    #WholeModel , word2vec: 7360, glove: 11390
    #save_dict_to_disk(wordId_wordEmbed_dict,"{}/word2vec_wordId_wordEmbed_dict_6More_train_val_2014.npy".format(dirname)) 
    #save_dict_to_disk(wordId_wordEmbed_dict,"{}/glove_wordId_wordEmbed_dict_6More_train_val_2014.npy".format(dirname)) 

    #TinyModel val//20,  word2vec: 1025, glove: 1116 
    #save_dict_to_disk(wordId_wordEmbed_dict,"{}/word2vec_wordId_wordEmbed_dict_6More_small_val_2014.npy".format(dirname)) 
    save_dict_to_disk(wordId_wordEmbed_dict,"{}/glove_wordId_wordEmbed_dict_6More_small_val_2014.npy".format(dirname)) 
    print "words in embed dict : {}".format(len(wordId_wordEmbed_dict.keys()))

def crop_img(img):
    if img.ndim == 2:
        img = img[:,:, np.newaxis]
        img = np.tile(img,(1,1,3))
    elif img.shape[2] == 4:
        img = img[:,:,:3]
    W = img.shape[0]
    H = img.shape[1]
    if W > H:
        H_new = 256
        W_new = int(W*H_new/H)
    else:
        W_new = 256
        H_new = int(H*W_new/W)
    I_new = resize(img,(W_new,H_new))  # [0,255] -> [0,1]
    I_crop = I_new[int(W_new/2)-112:int(W_new/2)+112,int(H_new/2)-112:int(H_new/2)+112,:]
    return I_crop

def save_imgFeature_imgFeatureId(dirname):
    data_dir = "/home/angeltop1/DLResearch/Google_Refexp_toolbox/external/coco"
    json_files = ["/home/angeltop1/DLResearch/Google_Refexp_toolbox/external/coco/annotations/captions_train2014.json",
                  "/home/angeltop1/DLResearch/Google_Refexp_toolbox/external/coco/annotations/captions_val2014.json"]
                  
    vgg_load_model ="/home/angeltop1/dummy/luafiles/dir_tryCNN/VGG19/VGG_ILSVRC_19_layers.caffemodel"
    func = caffe.CaffeFunction(vgg_load_model)
    imgId_imgFeatureId = {}
    featureId = 0
    print "load caffe model successfully"
    for json_file in json_files:
        print "processing file : {}".format(json_file)
        data = json.load(open(json_file,'r'))

        #build {image_id: image_file_name} index 
        imgId_to_imgFilename = {}
        prefix = json_file.split("/")[-1].split(".")[0].split("_")[1]
        print "prefix is {}".format(prefix)
        for i in range(len(data["images"])):
            idx = data["images"][i]["id"]
            filename = data["images"][i]["file_name"]
            imgId_to_imgFilename[idx] = filename
        print "build id_filename index successfully, {} items".format(len(imgId_to_imgFilename.keys()))

        #extract features 
        imgFeatureId_imgFeature = {}
        length = len(data["annotations"]) #val //20 For TinyModel
        for i in range(length):
            if i% 1000 == 0:
                print "{} image skipped, feature ID is {}".format(i,featureId)
            imgId = data["annotations"][i]["image_id"]
            if imgId in imgId_imgFeatureId:
                continue

            imgId_imgFeatureId[imgId] = featureId
            I = io.imread("{}/images/{}/{}".format(data_dir,prefix,imgId_to_imgFilename[imgId]))
            I_crop = crop_img(I)
            x_data = I_crop
            x_data = np.transpose(x_data,(2,0,1)).reshape(1,3,224,224)
            x = Variable(x_data.astype(np.float32))
            y, = func(inputs={'data':x},outputs=['conv5_4'])
            #y = Variable(np.array(np.random.random((1,512,14,14))).astype(np.float32))
            imgFeatureId_imgFeature[featureId] = y.data.reshape(512,14*14)
            featureId += 1
        #TinyModel 2122 Images, WholeModel {} Images,
        print "{} image solved !".format(featureId) 
        save_dict_to_disk(imgFeatureId_imgFeature,"{}/imgFeatureIdimgFeature_dict_6More_{}.npy".format(dirname,prefix)) 
    save_dict_to_disk(imgId_imgFeatureId,"{}/imgIdimgFeatureId_dict_6More_train_val_2014.npy".format(dirname))
   
def save_img_feature_to_file(dirname):
    #load caption file 
    data_dir = "/home/angeltop1/DLResearch/Google_Refexp_toolbox/external/coco"
    json_files = ["/home/angeltop1/DLResearch/Google_Refexp_toolbox/external/coco/annotations/captions_train2014.json",
                  "/home/angeltop1/DLResearch/Google_Refexp_toolbox/external/coco/annotations/captions_val2014.json"]

    word_id_dict = load_data_from_disk("{}/word_id_dict_6More_train_val_2014.npy".format(dirname)) 
    #word_id_dict = load_data_from_disk("{}/word_id_dict_6More_small_val_2014.npy".format(dirname))  #TinyModel val//20

    #same for TinyModel and Whole Model
    imgId_imgFeatureId = load_data_from_disk("{}/imgIdimgFeatureId_dict_6More_train_val_2014.npy".format(dirname))
    for json_file in json_files:
        data = json.load(open(json_file,'r'))
        #build {image_id: image_file_name} index 
        imgId_to_imgFilename = {}
        prefix = json_file.split("/")[-1].split(".")[0].split("_")[1]
        print "prefix is {}".format(prefix)
        for i in range(len(data["images"])):
            idx = data["images"][i]["id"]
            filename = data["images"][i]["file_name"]
            imgId_to_imgFilename[idx] = filename
        print "build id_filename index successfully, {} items".format(len(imgId_to_imgFilename.keys())) 

        #extract features
        train2014_data = []  
        length = len(data["annotations"]) #//20 ,TinyModel
        for i in range(length):
            sents = data["annotations"][i]["caption"].strip().split()
            new_sents = []
            length = len(sents)
            for ii, word in enumerate(sents):
                new_sents.append(regex.sub('',word))
                if re.search(regex,word):
                    if not word[0].isalpha():
                        new_sents.append(word[0]) 
                    if not word[-1].isalpha():    
                        new_sents.append(word[-1])
                else:
                    if ii == length - 1:
                        new_sents.append(".")

            word_ids = []  #<eos> 0 , UNK 1, <bos> 2,
            word_ids.append(2) # <bos>
            for word in new_sents:
                if word in word_id_dict:
                    word_ids.append(word_id_dict[word])
                else:
                    word_ids.append(1)
            word_ids.append(0)  #<eos>

            imgId = data["annotations"][i]["image_id"]
            featureId = imgId_imgFeatureId[imgId]
            train2014_data.append([word_ids,featureId])

        #TinyModel 10132 Training Instance ,Whole Model 414113,{} Training instance
        print "{} training datas!".format(len(train2014_data)) 
        save_dict_to_disk(train2014_data,"{}/{}_trainData.npy".format(dirname,prefix))   # Whole Model 202654 Val instance

def save_img_feature_to_file_valdata(dirname):
    #load caption file 
    #data_dir = "C:/Users/Ran Wensheng/Desktop/CaptionGenModelData"
    json_files = ["C:/Users/Ran Wensheng/Desktop/CaptionGenModelData/MSCOCO_data/annotations/captions_val2014.json",
                  #"C:/Users/Ran Wensheng/Desktop/CaptionGenModelData/MSCOCO_data/annotations/captions_train2014.json"
                  ]

    #json_files = ["MSCOCO_data/annotations/captions_val2014.json","MSCOCO_data/annotations/captions_train2014.json"]
    word_id_dict = load_data_from_disk("word_id_dict_4More_train2014.npy")
    imgId_imgFeatureId = load_data_from_disk("TrainVal2014_imgIdimgFeatureId_dict.npy")
    for json_file in json_files:
        data = json.load(open(json_file,'r'))
        #build {image_id: image_file_name} index 
        imgId_to_imgFilename = {}
        prefix = json_file.split("/")[-1].split(".")[0].split("_")[1]
        print "prefix is {}".format(prefix)
        for i in range(len(data["images"])):
            idx = data["images"][i]["id"]
            filename = data["images"][i]["file_name"]
            imgId_to_imgFilename[idx] = filename
        print "build id_filename index successfully, {} items".format(len(imgId_to_imgFilename.keys())) 

        #extract features
        length = len(data["annotations"]) #TinyModel
        imgId_sents = {}
        for i in range(length):
            sents = data["annotations"][i]["caption"].strip().split()
            new_sents = []
            length = len(sents)
            for ii, word in enumerate(sents):
                new_sents.append(regex.sub('',word))
                if re.search(regex,word):
                    if not word[0].isalpha():
                        new_sents.append(word[0]) 
                    if not word[-1].isalpha():    
                        new_sents.append(word[-1])
                else:
                    if ii == length - 1:
                        new_sents.append(".")

            word_ids = []  #<eos> 0 , UNK 1, <bos> 2
            word_ids.append(2)
            for word in new_sents:
                if word in word_id_dict:
                    word_ids.append(word_id_dict[word])
                else:
                    word_ids.append(1)
            
            word_ids.append(0)  #<eos>
            imgId = data["annotations"][i]["image_id"]
            if imgId not in imgId_sents:
                imgId_sents[imgId] = []
            imgId_sents[imgId].append(word_ids)

        train2014_data = []  
        length = len(data["images"])
        for i in range(length):
            imgId = data["images"][i]["id"]
            featureId = imgId_imgFeatureId[imgId]
            annotation_embeds = imgId_sents[imgId]
            train2014_data.append([annotation_embeds,featureId])            

        print "{} training datas!".format(len(train2014_data)) #Tiny Moel 50663 Training Instance
        save_dict_to_disk(train2014_data,"{}trainData.npy".format(prefix))


def save_npy_to_db(_dir):
    file_names = ["val"]
    for file_name in file_names:
        feat_dict = load_data_from_disk("{}/imgFeatureIdimgFeature_dict_6More_{}2014.npy".format(_dir,file_name))
        print "load {} successfully".format(file_name)
        save_data_to_db("{}/imgFeatureIdimgFeature_dict_6More_{}2014_db".format(_dir,file_name),feat_dict.values(),feat_dict.keys())
    
def score(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores


def write_hyper_to_file():
    TinyModel_model_hyper_setting = {}
    _dir = "../TinyModel"
    TinyModel_model_hyper_setting["e"] = 200
    TinyModel_model_hyper_setting["n1"] = 2
    TinyModel_model_hyper_setting["n2"] = 2
    TinyModel_model_hyper_setting["b"] = 64#100
    TinyModel_model_hyper_setting["h"] = 1800#1800
    TinyModel_model_hyper_setting["nw"] = 512
    TinyModel_model_hyper_setting["gpu"] = 1
    TinyModel_model_hyper_setting["lr"] = 1e-2#tensor 1e-3
    TinyModel_model_hyper_setting["d"] = 0.5
    TinyModel_model_hyper_setting["c1"] = 1.0
    TinyModel_model_hyper_setting["c2"] = 0.002
    TinyModel_model_hyper_setting["c3"] = 0.1
    TinyModel_model_hyper_setting["v"] = 1119
    TinyModel_model_hyper_setting["estop"] = 2000
    TinyModel_model_hyper_setting["trainVal"] = 20
    TinyModel_model_hyper_setting["p"] = 50
    TinyModel_model_hyper_setting["train_data"] = "{}/val2014_trainData.npy".format(_dir)
    TinyModel_model_hyper_setting["val_data"] = "{}/val2014_trainData.npy".format(_dir)
    TinyModel_model_hyper_setting["id_embed"] = "{}/glove_wordId_wordEmbed_dict_6More_small_val_2014.npy".format(_dir)
    TinyModel_model_hyper_setting["train_id_feat"] = "{}/imgFeatureIdimgFeature_dict_6More_val2014_db".format(_dir)
    TinyModel_model_hyper_setting["val_id_feat"] = "{}/imgFeatureIdimgFeature_dict_6More_val2014_db".format(_dir)
    #TinyModel_model_hyper_setting["train_id_feat"] = "{}/imgFeatureIdimgFeature_dict_6More_val2014.npy".format(_dir)
    #TinyModel_model_hyper_setting["val_id_feat"] = "{}/imgFeatureIdimgFeature_dict_6More_val2014.npy".format(_dir)
    TinyModel_model_hyper_setting["word_id"] = "{}/word_id_dict_6More_small_val_2014.npy".format(_dir)
    TinyModel_model_hyper_setting["img_id"] = "{}/imgIdimgFeatureId_dict_6More_train_val_2014.npy".format(_dir)
    TinyModel_model_hyper_setting["jsonFile"] = "/home/angeltop1/DLResearch/Google_Refexp_toolbox/external/coco/annotations/captions_val2014.json"    #for evaluation
    save_dict_to_disk(TinyModel_model_hyper_setting,"{}/model_hyper_setting".format(_dir))

    _dir = "../WholeModel"
    TinyModel_model_hyper_setting["e"] = 50
    TinyModel_model_hyper_setting["n1"] = 2
    TinyModel_model_hyper_setting["n2"] = 2
    TinyModel_model_hyper_setting["b"] = 64
    TinyModel_model_hyper_setting["h"] = 1800
    TinyModel_model_hyper_setting["nw"] = 512#300
    TinyModel_model_hyper_setting["gpu"] = 2
    TinyModel_model_hyper_setting["lr"] = 1e-3#1e-2
    TinyModel_model_hyper_setting["d"] = 0.5
    TinyModel_model_hyper_setting["c1"] = 1.0
    TinyModel_model_hyper_setting["c2"] = 0.002
    TinyModel_model_hyper_setting["c3"] = 0.1
    TinyModel_model_hyper_setting["v"] = 10503
    TinyModel_model_hyper_setting["estop"] = 2000
    TinyModel_model_hyper_setting["trainVal"] = 10000
    TinyModel_model_hyper_setting["p"] = 10
    TinyModel_model_hyper_setting["train_data"] = "{}/train2014_trainData.npy".format(_dir)
    TinyModel_model_hyper_setting["val_data"] = "{}/val2014_trainData.npy".format(_dir)
    TinyModel_model_hyper_setting["id_embed"] = "{}/glove_wordId_wordEmbed_dict_6More_train_val_2014.npy".format(_dir)
    TinyModel_model_hyper_setting["train_id_feat"] = "{}/imgFeatureIdimgFeature_dict_6More_train2014_db".format(_dir)
    TinyModel_model_hyper_setting["val_id_feat"] = "{}/imgFeatureIdimgFeature_dict_6More_val2014_db".format(_dir)
    #TinyModel_model_hyper_setting["train_id_feat"] = "{}/imgFeatureIdimgFeature_dict_6More_train2014.npy".format(_dir)
    #TinyModel_model_hyper_setting["val_id_feat"] = "{}/imgFeatureIdimgFeature_dict_6More_val2014.npy".format(_dir)
    TinyModel_model_hyper_setting["word_id"] = "{}/word_id_dict_6More_train_val_2014.npy".format(_dir)
    TinyModel_model_hyper_setting["img_id"] = "{}/imgIdimgFeatureId_dict_6More_train_val_2014.npy".format(_dir)
    TinyModel_model_hyper_setting["jsonFile"] = "/home/angeltop1/DLResearch/Google_Refexp_toolbox/external/coco/annotations/captions_val2014.json"    #for evaluation
    save_dict_to_disk(TinyModel_model_hyper_setting,"{}/model_hyper_setting".format(_dir))

