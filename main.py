#!/usr/bin/python
# -*- coding: utf-8 -*-

# please set paths to the required tools in config.py

# this scripts performs (part of) the experiments described in (Sennrich, Schwenk & Aransa 2013).
# Specifically, it takes a vector of (source-side) texts (LM_TEXTS), trains a language model on each,
# and then clusters a development set (DEV_L1/DEV_L2) with K-means clustering (K clusters).

# for each cluster, the set of phrase pairs is extracted with MOSES_TRAINING,
# given a moses config (MOSES_CFG) that uses a PhraseDictionaryMultiModelCounts0 with all desired component models,
# the script starts a moses server instance (MOSES_SERVER, MOSES_SERVER_PORT, MOSES_SERVER_URL) and optimizes the translation model weights for each cluster.

# finally, the script translates the test set (TEST_SET) by assigning each test set sentence to the closest cluster, then using its weights for translation.

# What this script does *not* do (but what is described in (Sennrich, Schwenk & Aransa 2013), is a retuning of the log-linear parameters (MERT) with the optimized translation models,
# and language model switching.



import sys
import os
import shutil
import gzip
from subprocess import Popen, PIPE

#directory which contains scripts (hardcode this if you use PBS or other cluster infrastructure which moves scripts)
sys.path.append('./')
import cluster
import translate

from config import *


def create_clusters():
    """create K clusters from development data, and store relevant information to persistent_data.txt"""

    c = cluster.Cluster(LMs, DEV_L1, DEV_L2, K, general_lm=GENERAL_LM, working_dir=WORKING_DIR)
    clusters, centroids = c.kmeans()
    c.writedown(clusters)
    c.write_persistent_data(clusters, centroids, 'persistent_data.txt')

    return centroids


def optimize_weights_online():
    """optimize instance weights on a set of phrase pairs. uses mosesserver, so no system restart is required"""

    weights = []

    for i in range(K):
        temp_model_path = os.path.join(WORKING_DIR, 'model' + str(i))
        bitext_path = os.path.join(WORKING_DIR, str(i))

        specific_options = ['-root-dir', temp_model_path,
                            '-corpus', bitext_path,
                            '-f', 's', '-e', 't']

        train_cmd = MOSES_TRAINING + specific_options
        p = Popen(train_cmd)
        p.wait()

        extract_file = os.path.join(temp_model_path, 'model', 'extract.sorted.gz')
        phrase_pairs = read_phrase_pairs(gzip.open(extract_file))

        weight_flat = translate.optimize(phrase_pairs, MOSES_SERVER_URL)
        weights.append(weight_flat)

    sys.stderr.write('All weights:\n')
    for w in weights:
        sys.stderr.write(' '.join(map(str,w)) + '\n')
    sys.stderr.write('\n')

    return weights


def read_phrase_pairs(input_object):
    """convert Moses extract file into a list of phrase pairs"""

    pb = []
    for line in input_object:
        line = line.split(' ||| ')
        pb.append((line[0],line[1]))
    return pb


def translate_text(text, centroids, weights, output_file):
    """translate a text with given centroids and instance weights for each centroid.
       each sentence is assigned to the closest centroid and during translation, the corresponding instance weights are used"""

    sent_in = open(text,'r').readlines()
    C = cluster.Cluster(LMs, None, None, K, general_lm=GENERAL_LM, working_dir=WORKING_DIR)
    weighted_text = translate.assign_weights(sent_in, C, centroids, weights)
    translate.translate_concurrent(weighted_text, MOSES_SERVER_URL, output_file, NUM_PROCESSES)


def write_weights_to_file(weights, f):

    fobj = open(os.path.join(WORKING_DIR,f),'w')
    for w in weights:
        fobj.write(' '.join(map(str,w)) + '\n')
    fobj.close()


if __name__ == '__main__':

    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)

    # copy files to working directories so you can re-use config
    shutil.copy(sys.argv[0], WORKING_DIR)
    shutil.copy('cluster.py', WORKING_DIR)
    shutil.copy('translate.py', WORKING_DIR)
    shutil.copy('config.py', WORKING_DIR)


    # train LMs
    GENERAL_LM = None
    LMs = []
    if USING_LM_PATHS:
        if GENERAL_LM_TEXT:
            GENERAL_LM = cluster.SRILM_interface(GENERAL_LM_TEXT, order=LM_ORDER)
        LMs = [cluster.SRILM_interface(f, order=LM_ORDER) for f in LM_TEXTS]
    else:
        if GENERAL_LM_TEXT:
            lm_name = os.path.basename(GENERAL_LM_TEXT).split('.')[0] + '.lm'
            lm_name = os.path.join(LM_DIR, lm_name)
            GENERAL_LM = cluster.SRILM_interface(lm_name, order=LM_ORDER, text=GENERAL_LM_TEXT)

        for textfile in LM_TEXTS:
            lm_name = os.path.basename(textfile).split('.')[0] + '.lm'
            lm_name = os.path.join(LM_DIR, lm_name)
            LMs.append(cluster.SRILM_interface(lm_name, order=LM_ORDER, text=textfile))

    sys.stderr.write('Executing: ' + ' '.join(MOSES_SERVER) + '\n')
    p = Popen(MOSES_SERVER) # we start server first because it needs to be up when we start translating.

    # if not already done, cluster development data
    if os.path.exists(os.path.join(WORKING_DIR, 'persistent_data.txt')):
        centroids = translate.read_centroids(os.path.join(WORKING_DIR, 'persistent_data.txt'))
    else:
        centroids = create_clusters()

    # if not already done, optimize instance weights for each cluster
    if os.path.exists(os.path.join(WORKING_DIR, 'persistent_weights.txt')):
        weights = translate.read_weights(os.path.join(WORKING_DIR, 'persistent_weights.txt'))
    else:
        weights = optimize_weights_online()
        write_weights_to_file(weights, 'persistent_weights.txt')

    # translate a new text
    translate_text(TEST_SET, centroids, weights, os.path.join(WORKING_DIR, 'output.txt'))
    p.kill()
