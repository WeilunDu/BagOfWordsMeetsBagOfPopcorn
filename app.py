#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
import tornado
from tornado.ioloop import IOLoop
from tornado import web, gen, process, httpserver, httpclient, netutil
import pickle
import logging
import urllib
from itertools import chain
from collections import *
import json
import numpy as np
from settings import dmLabeled, dbowLabeled, trainedClassifer, BASE_PORT, servers
from doc2vec import Doc2Vec
from doc2vec import LabeledSentence
from docVecTrain import flushLoggerInfo, cleanUpText
from sklearn.linear_model import SGDClassifier

#load trained classifer
if os.path.isfile(trainedClassifer):
    model_lr = pickle.load(open(trainedClassifer, 'rb')) 
else:
    raise RuntimeError("Must first train classsifer model") 


SETTINGS = {"static_path": "./webapp"}


'''
A simple web interface that accepts a movie review of any length and return whether 
it is positive or negative.

'''

class Doc2vecServer(web.RequestHandler):
    def initialize(self, model):
        self._model = model

    def get(self):
        review = self.get_argument('review', None)
        if review is None:
            return
        self.write(json.dumps(list(self._model.train_online(cleanUpText(review)))))
        

class Web(web.RequestHandler):

    @gen.coroutine
    def get(self):
        review = self.get_argument('review', None).encode('utf-8')
        if review is None:
            return
        global model_lr
        
       
        #fetch document vector from doc2vec servers
        http = httpclient.AsyncHTTPClient()
        print servers['doc2vec']
        responses = yield [ http.fetch("http://%s/doc2vec?%s" %\
                     (server, urllib.urlencode({'review': review})))\
                      for server in servers['doc2vec']]
        
        vecs = ()
        for r in responses:
            vecs = vecs + (json.loads(r.body),)
        
        test_vec = np.hstack(vecs)
        
        self.write(json.dumps({"result" : model_lr.predict([test_vec])[0]}))
        self.finish()


#this is a very typical way to start multiple server processes
def main():
    numProcs = 3
    taskID = process.fork_processes(numProcs, max_restarts=0)
    port = BASE_PORT + taskID
    if taskID == 0:
        app = httpserver.HTTPServer(tornado.web.Application([
            (r"/submit", Web)], **SETTINGS))
        logging.info("webapp listening on %d" % port)
    else:
        #load trained model from either dm or dbow
        if os.path.isfile(dmLabeled) and os.path.isfile(dbowLabeled):
            fname = dmLabeled if taskID == 1 else dbowLabeled
            model = Doc2Vec.load(fname)   
        else:
            raise RuntimeError("Must first train doc2vec model")

        app = httpserver.HTTPServer(web.Application([(r"/doc2vec", Doc2vecServer, dict(model = model))]))
        logging.info("Doc2vec server %d listening on %d" % (taskID, port))
    
    app.add_sockets(netutil.bind_sockets(port))
    IOLoop.current().start()

if __name__ == "__main__":
    
    logging.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s', level=logging.DEBUG)
    main()




