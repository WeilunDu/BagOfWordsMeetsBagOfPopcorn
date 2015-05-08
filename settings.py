import hashlib
#hyperparameter for doc2vec
size = 400

#file paths
useModifiedModule = True
dmUnlabeled = "data/dm_unlabeled_%d.doc2vec" % size
dbowUnlabeled = "data/dbow_unlabeled_%d.doc2vec" % size
dmLabeled = 'data/modified_dm_labeled_%d.doc2vec' % size
dbowLabeled = 'data/modified_dbow_labeled_%d.doc2vec' % size
expDmLabeled = 'data/exp_dm_labeled_%d.doc2vec' % size
expDbowLabeled = 'data/exp_dbow_labeled_%d.doc2vec' % size
testRes = 'data/word2vec_sgd_lr_%d.csv' % size
trainedClassifer = 'data/sgd_lr_%d.pkl' % size
MAX_PORT = 49152
MIN_PORT = 10000
#port number
BASE_PORT = int(hashlib.md5("wd387").hexdigest()[:8], 16) % \
  (MAX_PORT - MIN_PORT) + MIN_PORT

servers = {}
servers['doc2vec'] = ["127.0.0.1:%d" % port for port in range(BASE_PORT + 1, BASE_PORT + 3)]
