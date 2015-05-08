#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
from settings import size, dmUnlabeled, dbowUnlabeled,\
 dmLabeled, dbowLabeled, useModifiedModule, testRes, trainedClassifer
import logging
import numpy as np
import pickle
from sklearn.linear_model import SGDClassifier
from doc2vec import Doc2Vec
from doc2vec import LabeledSentence
from docVecTrain import flushLoggerInfo, cleanUpText

if __name__ == "__main__":
    
    flushLoggerInfo()
    if os.path.isfile(dmLabeled) and os.path.isfile(dbowLabeled)\
       and os.path.isfile(trainedClassifer):
        model_dm = Doc2Vec.load(dmLabeled)   
        model_dbow = Doc2Vec.load(dbowLabeled)
        neg = "This movie is an absolute disaster within a disaster film. It is full of great action scenes, which are only meaningful if you throw away all sense of reality. Let's see, word to the wise, lava burns you; steam burns you. You can't stand next to lava. Diverting a minor lava flow is difficult, let alone a significant one. Scares me to think that some might actually believe what they saw in this movie.<br /><br />Even worse is the significant amount of talent that went into making this film. I mean the acting is actually very good. The effects are above average. Hard to believe somebody read the scripts for this and allowed all this talent to be wasted. I guess my suggestion would be that if this movie is about to start on TV ... look away! It is like a train wreck: it is so awful that once you know what is coming, you just have to watch. Look away and spend your time on more meaningful content."
        neg_test_vecs = np.hstack((model_dm.train_online(cleanUpText(neg)), model_dbow.train_online(cleanUpText(neg))))
        pos = "Naturally in a film who's main themes are of mortality, nostalgia, and loss of innocence it is perhaps not surprising that it is rated more highly by older viewers than younger ones. However there is a craftsmanship and completeness to the film which anyone can enjoy. The pace is steady and constant, the characters full and engaging, the relationships and interactions natural showing that you do not need floods of tears to show emotion, screams to show fear, shouting to show dispute or violence to show anger. Naturally Joyce's short story lends the film a ready made structure as perfect as a polished diamond, but the small changes Huston makes such as the inclusion of the poem fit in neatly. It is truly a masterpiece of tact, subtlety and overwhelming beauty."
        pos_test_vecs = np.hstack((model_dm.train_online(cleanUpText(pos)), model_dbow.train_online(cleanUpText(pos))))
        with open(trainedClassifer, 'rb') as f:
            lr = pickle.load(f)
            #we are expecting the classification to be 0 but there is no guarantee 
            print lr.predict([neg_test_vecs, pos_test_vecs]) 
    else:
        print 'runtime error'
        sys.exit(1)
