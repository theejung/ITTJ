import os, sys, argparse,nltk
import numpy as np
import tensorflow as tf

from utils import confusion_matrix, prec_rec, PaddingDatawithTarget, load_glove, load_embedding
from data_loader import MRDataLoader, S140DataLoader



class SentiDict():
  def __init__(self,args):

    self.num_max_epochs = 1
    self.dataset = args.dataset

    # Load data
    data_loader = None
    if args.dataset == 'moviereview': data_loader = MRDataLoader
    elif args.dataset == 'senti140': data_loader = S140DataLoader
    else: print 'wrong data'; sys.exit(1)

    loader = data_loader(
        data_path = '../data/%s/'%(args.dataset))
    loader.read_data_text(only_test=True)

    # Data iterators
    #NOTE no need training data for sentiment dictionary matching
    #self.train_iter = PaddingDatawithTarget(loader.train)
    self.test_iter = PaddingDatawithTarget(loader.test)

  def test(self, verbose=False):
    # labels
    # moviereview: 0,1,2,3,4
    # senti140: 0,2,4 (ignore 2)



    correct, incorrect = 0, 0
    while self.test_iter.epochs < self.num_max_epochs:
      sent,label,_ = self.test_iter.next_batch_text(1)
      label = int(label)
      if self.dataset == 'senti140':
        if label == 2: continue # ignore neutral score

      score_pos,score_neg = 0, 0
      for word in nltk.word_tokenize(sent):
        if word in self.senti_dict['positive']:
          score_pos += 1
        if word in self.senti_dict['negative']:
          score_neg += 1
      score = score_pos - score_neg
      #print score, label
      #import pdb; pdb.set_trace()


      if self.dataset == 'senti140':
        if (score >=0 and label == 4) or (score<0 and label == 2): correct += 1
        else: incorrect += 1
      if self.dataset == 'moviereview':
        if (score ==0 and label == 2) \
            or (score==1 and label == 3) or (score>1 and label == 4) \
            or (score==-1 and label == 1) or (score<-1 and label ==0):
              correct += 1
        else: incorrect += 1


    print 'Test accuracy: %.2f (%d/%d)'%(correct*100./(correct+incorrect), correct, incorrect)

  def build_senti_dict(self, senti_dict_filename = "subjclueslen1-HLTEMNLP05.tff"):
    # Build the lexicon...
    senti_dict = {}
    senti_list = ['positive','negative']
    for senti in senti_list:
      senti_dict[senti] = {}
    for line in open(senti_dict_filename,'r'):
      try:
          mytype,mylen,word,pos,stemmed,prior = line.strip().split(' ')
      except:
          continue
      word = word.split('=')[-1]
      senti = prior.split('=')[-1]
      if senti in senti_list:
          senti_dict[senti][word] = 1
      self.senti_dict = senti_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default = 'moviereview') #'senti140'
    args = parser.parse_args()

    model = SentiDict(args)
    model.build_senti_dict( senti_dict_filename = "subjclueslen1-HLTEMNLP05.tff")
    model.test()




