# -*- coding: utf-8 -*-
# file paths etc. need to be set in this file for the demo to work

import os

# path to Moses binary directory
MOSES_BIN = '/home/rico/moses-git/bin'

#directory in which intermediate files (and final output) are written.
WORKING_DIR = 'demo/working_dir'

#directory which contains GIZA++; will be used as argument of '-external-bin-dir' in Moses' train-model.perl
GIZA_DIR = '/home/rico/bin/'


# If this is set to true, LM_TEXTS and GENERAL_LM_TEXT are not interpreted as (zipped) text files, but as trained LMs
USING_LM_PATHS = False

LM_TEXTS =  ['demo/train1.de'
            ,'demo/train2.de'
            ]

LM_ORDER = 4

# if using USING_LM_PATHS is False (and new language models are trained from text files), put them in this directory
LM_DIR = WORKING_DIR


DEV_L1 = 'demo/dev.de'
DEV_L2 = 'demo/dev.en'

TEST_SET = 'demo/test.de'


# number of clusters in k-means clustering
K = 2

# number of processes used by Moses server/client
NUM_PROCESSES = 4


#Path of Moses config. Should use PhraseDictionaryMultiModelCounts phrase table with pre-trained component models 
#(check the Moses documentation on how to use MultiModelCounts phrase table type, and how to prepare the component models: http://www.statmt.org/moses/?n=Moses.AdvancedFeatures)
MOSES_CFG = 'demo/moses.ini'

MOSES_SERVER_PORT = '8111'

#where the moses server can be reached
MOSES_SERVER_URL = "http://localhost:{0}/RPC2".format(MOSES_SERVER_PORT)




## settings for which default should work, but which you may want to override ##

# path to KenLM query tool; is distributed with Moses
QUERY_CMD = os.path.join(MOSES_BIN, 'query')

# path to Ken's lm-plz tool; is distributed with Moses
LMPLZ_CMD = os.path.join(MOSES_BIN, 'lmplz')

# path to mosesserver; is distributed with Moses
SERVER_CMD = os.path.join(MOSES_BIN, 'mosesserver')

# path to train-model.perl script; is distributed with Moses
MOSES_TRAIN_CMD = os.path.join(MOSES_BIN, '..', 'scripts', 'training', 'train-model.perl')

#location of parameters that are used to start Moses server
MOSES_SERVER = [SERVER_CMD,
    '-f', MOSES_CFG, 
    '-use-persistent-cache', '0', 
    '--server-port', MOSES_SERVER_PORT]

#Used for word alignment and phrase extraction on each cluster (we need list of phrase pairs for translation model weight optimization)
MOSES_TRAINING = [MOSES_TRAIN_CMD,
    '-external-bin-dir', GIZA_DIR,
    '-alignment', 'grow-diag-final-and', 
    '--write-lexical-counts', 
    '--last-step=5',
    '-parallel', 
    '--cores', '4', 
    '-lm', '0:5:/etc/passwd', #we don't need an LM, but the moses training scripts requires an existing, non-empty file
    '-sort-buffer-size', '20G']

# Point to a general domain text file to use cross-entropy difference between LM and general LM instead of just LM cross-entropy for k-means clustering. (Moore & Lewis 2010)
# Not used in (Sennrich, Schwenk and Aransa 2013) because cosine works well with LM cross-entropy.
GENERAL_LM_TEXT = ''