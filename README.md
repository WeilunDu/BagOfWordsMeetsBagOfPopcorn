# BagOfWordsMeetsBagOfPopcorn
############################################################

To evaluate this project please first make sure 
that you have installed
sklearn, gensim, beautifulSoup, cython

python setup.py build_ext --inplace #to build the c-extension


python docVecTrain.py #to build the doc2vec models and classifier, it will take roughly 30 mins





python app.py# to run the web app

Due to the limited timing, I didn't have a front-end.
You may enter the url into the browser
"http://[domain]:39179/submit?review="MOVIE REVIEW FROM TEST SET WOULD BE PREFERED"

a json will be returned such that 1 indicates positve feeling and 0 indicates negative feeling

The best score I have achieved in Kaggle competition
(without telling you the answer for test cases)
is 85.7% accuracy.
You can find this information at 
https://www.kaggle.com/c/word2vec-nlp-tutorial/leaderboard
by searching "Weilun Du" 


##############################################################
