
import os, sys, argparse, glob
import time
import numpy as np
from utils import load_embedding

class ContextExtract():
  def __init__(self,args):

    self.out_path = args.out_path
    self.data_path = args.data_path
    self.target = args.target

  def extract(self):

    files = sorted(glob.glob(self.data_path + '/tweets.*.out'))
    for fidx,file in enumerate(files):
      print '[%d/%d] reading... %s'%(fidx,len(files),file)
      time = file.split('/')[-1].split('.')[1]
      if not os.path.exists(os.path.join(self.out_path, self.target)):
        os.makedirs(os.path.join(self.out_path, self.target))
      fout = open(os.path.join(self.out_path, self.target, '%s.tweets'%(time)), 'w')

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
          fout.write(' '.join(words)+'\n')
      fout.close()


if __name__ == '__main__':

  # AAPL, AMZN, MSFT, FB, GOOG
  companies = ['amazon', 'apple', 'microsoft', 'facebook', 'google']
  for company in companies:
    print 'CONTEXT Extracting....',company
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default = '/data/cgraph/tweets/daily.1m.output/')
    parser.add_argument("--out_path", default = '/data/time2vec/tweets/')
    parser.add_argument("--target", default = 'amazon')
    args = parser.parse_args()
    args.target = company
    model = ContextExtract(args)
    model.extract()

