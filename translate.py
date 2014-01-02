#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Rico Sennrich <sennrich@cl.uzh.ch>

import sys
import time
import random
import cluster
from socket import error as SocketError
from multiprocessing import Pool

if sys.version_info < (3, 0):
    import xmlrpclib
else:
    import xmlrpc.client as xmlrpclib

#translate text using num_processes concurrent client processes (ideally the same as the number of server threads)
#each sentence can be given a different weight
#text is list of triples (line, weight, server_url).
def translate_concurrent(text, server_url, output_file, num_processes=8):

    out = open(output_file, 'w')

    pool = Pool(processes=num_processes)
    text_args = [(line, weight, server_url) for (line, weight) in text]

    for line in pool.imap(translate_single_line, text_args):
        out.write(line)


def translate_single_line(args):

    line, weights, server_url = args
    server = xmlrpclib.ServerProxy(server_url)

    params = {'text':line}
    params['lambda'] = weights
    params['model_name'] = 'PhraseDictionaryMultiModelCounts0'
    try:
        translation = server.translate(params)['text']
    # assume that socket errors mean that the model hasn't finished loading; if something else is wrong with server (e.g. port busy), program is stuck
    except SocketError:
        while True:
            time.sleep(10)
            try:
                translation =  server.translate(params)['text']
                break
            except SocketError:
                continue

    return translation.encode('utf-8') + '\n'


def optimize(phrase_pairs, url):

    server = xmlrpclib.ServerProxy(url)

    params = {'phrase_pairs':phrase_pairs,
              'model_name':'PhraseDictionaryMultiModelCounts0'}
    try:
        weights = server.optimize(params)
    # assume that socket errors mean that the model hasn't finished loading; if somethign else is wrong with server (e.g. port busy), program is stuck
    except SocketError:
        while True:
            time.sleep(10)
            try:
                weights = server.optimize(params)
                break
            except SocketError:
                continue

    # make sure we get valid weights back, and not error message.
    try:
        float(weights[0])
    except:
        raise RuntimeError('Error: Perplexity minimization requires Moses to be compiled with dlib (compilation option --with-dlib))')

    return weights


def assign_weights(text, Clusterer, centroids, weights):
    scores = Clusterer.score_lms(Clusterer.lms, '\n'.join(text))
    clusters, total_distance = Clusterer.assign(centroids, scores)
    weighted_text = {}

    for c in clusters:
        for sent in clusters[c]:
            weighted_text[sent] = weights[c]

    return [(text[i], weighted_text[i]) for i in range(len(text))]



def read_centroids(persistent_data_file):
    on = 0
    centroids = {}
    for line in open(persistent_data_file):
        if line.startswith('Centroids:'):
            i = 0
            on = 1
        elif on:
            if line == '\n':
                break
            centroid = map(float,line.split())
            centroids[i] = dict([(j,centroid[j]) for j in range(len(centroid))])
            i += 1
    return centroids


def read_weights(persistent_weights_file):
    weights = []
    for line in open(persistent_weights_file):
        weights.append(map(float,line.split()))

    return weights


if __name__ == '__main__':

    url = "http://localhost:8111/RPC2"

    sent_in = sys.stdin.readlines()
    C = cluster.Cluster(lms, None, None, None, f_distance=cosine, maximize = True, general_lm=general_lm)
    centroids = read_centroids('persistent_data_file.txt')
    weights = [[random.random() for i in range(2*4)] for j in range(len(centroids))] #assumes two translation models for demo purposes
    weighted_text = assign_weights(sent_in, C, centroids, weights)
    translate_concurrent(weighted_text, url, 'output.txt')
