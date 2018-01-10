
import os, sys, argparse, glob
import time
import numpy as np
from utils import load_embedding

class ContextEncoder():
  def __init__(self,args):

    self.out_path = args.out_path
    self.data_path = args.data_path
    self.target = args.target

    self.emb = load_embedding(
        emb_path = '/data/word2vec/',
        emb_filename= 'glove.42B.300d.w2v.bin')

    self.emb_size = len(self.emb['the'])

  def encode(self):

    fout = open(self.out_path+'%s.context'%(self.target),'w')

    files = sorted(glob.glob(self.data_path + '/tweets.*.out'))
    for fidx,file in enumerate(files):
      print '[%d/%d] reading... %s'%(fidx,len(files),file)
      time = file.split('/')[-1].split('.')[1]

      tweet_embs = []
      with open(file) as fin:
        for lidx, line in enumerate(fin):
          if lidx == 0 : continue
          tks = line.strip().split('\t')
          try:
            words = tks[4].lower().split()
          except Exception as e:
            print e
            continue

          if self.target not in words: continue

          tweet_emb = [] #np.zeros(self.emb_size)
          for word in words:
            if word in self.emb:
              tweet_emb.append(self.emb[word])
          if len(tweet_emb) ==0: continue
          tweet_emb = np.average(tweet_emb,axis=0)
          tweet_embs.append(tweet_emb)

          # if lidx%1000 == 0:
           #  print '\t[%d] lines in file'%(lidx)
      context_emb = np.average(tweet_embs, axis=0)
      print '\t',time, len(tweet_embs), context_emb.shape
      fout.write('%s\t%s\n'%(time, ' '.join(['%.5f'%(v) for v in context_emb])))

    fout.close()

if __name__ == '__main__':

  # AAPL, AMZN, MSFT, FB, GOOG
  companies = ['amazon', 'apple', 'microsoft', 'facebook', 'google']
  for company in companies:
    print 'CONTEXT ENCODING....',company
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default = '/data/tweets/daily.1m.output/')
    parser.add_argument("--out_path", default = '/data/time2vec/')
    parser.add_argument("--target", default = 'amazon')
    args = parser.parse_args()
    args.target = company

    model = ContextEncoder(args)
    model.encode()

