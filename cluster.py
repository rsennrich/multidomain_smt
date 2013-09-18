#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Rico Sennrich <sennrich@cl.uzh.ch>

from __future__ import division
import sys
import os
import random
import math
import operator
from subprocess import Popen, PIPE
from collections import defaultdict

from config import LMPLZ_CMD

def euclidean_distance(v1, v2):

    total = 0
    for dim in v1:
        total += (v1[dim] - v2[dim])**2
    total **= 1/len(v1)

    return total


def dot_product(v1,v2):

    dp = 0
    for dim in v1:
        if dim in v2:
            dp += v1[dim]*v2[dim]
    return dp


def cosine(v1,v2):

    try:
        return dot_product(v1,v2) / math.sqrt(dot_product(v1,v1)*dot_product(v2,v2))
    except ZeroDivisionError:
        return 0


# Distance between vector and centroid: implemented: cosine, euclidean_distance
DISTANCE_FUNCTION = cosine
#needs to correspond to distance function; maximize for cosine, minimize for euclidean_distance
MAXIMIZE = True

class Cluster(object):

    def __init__(self, lms, textfile_s, textfile_t, num_clusters, goldclusters = None, f_distance=DISTANCE_FUNCTION, maximize = MAXIMIZE, general_lm = None, working_dir = ''):
        if textfile_s:
            self.text_s = open(textfile_s).readlines()
            self.n = len(self.text_s)
        if textfile_t: 
            self.text_t = open(textfile_t).readlines()
        self.num_clusters = num_clusters
        self.goldclusters = goldclusters
        self.lms = lms
        self.f_distance = f_distance
        self.general_lm = general_lm
        self.working_dir = working_dir

        if maximize:
            self.neutral = float('-inf')
            self.better_than = operator.gt
        else:
            self.neutral = float('inf')
            self.better_than = operator.lt


    def score_lms(self, lms, text):
        scores = defaultdict(dict)

        # cross-entropy difference according to Moore & Lewis 2010
        if self.general_lm:
            general_scores = self.general_lm.get_perplexity(text)

        for i, lm in enumerate(lms):
            lm_scores = lm.get_perplexity(text)
            for j, score in enumerate(lm_scores):
                scores[j][i] = score
                if self.general_lm:
                    scores[j][i] -= general_scores[j]

        return scores


    def kmeans(self):

        scores = self.score_lms(self.lms, '\n'.join(self.text_s))

        centroids = self.random_centroids(scores)

        total_distance = self.neutral
        i = 0
        while total_distance == self.neutral or self.better_than(total_distance,old_total_distance):
            old_total_distance = total_distance
            clusters, total_distance = self.assign(centroids, scores)
            centroids = self.calc_centroids(clusters, scores)
            sys.stderr.write('Iteration {0}\n'.format(i))
            sys.stderr.write('Avg. distance/similarity to centroids: {0}\n'.format(total_distance/self.n))
            if self.goldclusters:
                entropy = self.calc_gold_entropy(clusters)
                sys.stderr.write('gold entropy: {0}\n'.format(entropy))
            i += 1

        return clusters, centroids


    def assign(self, centroids, scores):
        """expectation step: given centroids, assign each sentence to closest cluster"""

        clusters = defaultdict(set)

        total_distance = 0

        for sentence, vector in scores.items():
            best = self.neutral
            bestcluster = None
            for c, centroid in centroids.items():
                d = self.f_distance(centroid, vector)
                if self.better_than(d, best):
                    bestcluster = c
                    best = d

            if not bestcluster is None:
                clusters[bestcluster].add(sentence)
                total_distance += best
            else:
                sys.stderr.write('No cluster found (why???)\n')

        return clusters, total_distance


    def calc_distance(self, clusters, scores):
        """keep clusters as they are, recalculate centroids and distance of each data point to centroid"""

        centroids = self.calc_centroids(clusters, scores)

        total_distance = 0

        for c in clusters:
            for sentence in clusters[c]:
                vector = scores[sentence]
                total_distance += self.f_distance(centroids[c], vector)

        return total_distance


    def random_centroids(self, scores):
        """random initialisation of centroids"""
        sample = random.sample(scores.keys(), self.num_clusters)

        centroids = {}
        for i in range(self.num_clusters):
            centroids[i] = scores[sample[i]]

        return centroids

    def calc_centroids(self, clusters, scores):
        """maximization step: calculate centroids from cluster members"""

        centroids = {}

        for c in clusters:
            centroid = defaultdict(float)
            for sentence in clusters[c]:
                for feature, value in scores[sentence].items():
                    centroid[feature] += value

            for feature in centroid:
                centroid[feature] /= len(clusters[c])

            centroids[c] = centroid

        return centroids


    def calc_gold_entropy(self, clusters):
        """given a set of true (gold) clusters, calculate entropy (the lower, the more similar the unsupervised clusters are to the gold clusters)"""

        entropy = 0

        for c in clusters:
            entropy_cluster = 0
            for gc in self.goldclusters.values():
                prob = len(gc.intersection(clusters[c])) / len(clusters[c])
                if prob:
                    entropy_cluster += -prob*math.log(prob,2)

            entropy += entropy_cluster * len(clusters[c])/self.n

        return entropy

    def writedown(self, clusters):
        for i in range(self.num_clusters):
            out_s = open(os.path.join(self.working_dir,"{0}.s".format(i)),'w')
            out_t = open(os.path.join(self.working_dir,"{0}.t".format(i)),'w')

            for sentence in clusters[i]:
                out_s.write(self.text_s[sentence])
                out_t.write(self.text_t[sentence])

            out_s.close()
            out_t.close()


    def write_persistent_data(self, clusters, centroids, f):
        """write some statistics to file for later re-use (LM paths, config options, centroids, which sentence is assigned to which cluster)"""
        fobj = open(os.path.join(self.working_dir,f),'w')
        fobj.write('LMs:\n')
        for lm in self.lms:
            fobj.write(lm.name + '\n')
        fobj.write('\n')

        if self.general_lm:
            fobj.write('General_LM:\n' + self.general_lm.name + '\n\n')

        fobj.write('Distance:\n' + self.f_distance.__name__ + '\n\n')

        if self.better_than == operator.gt:
            maximize = '1'
        else:
            maximize = '0'
        fobj.write('Maximize:\n' + maximize + '\n\n')

        fobj.write('Centroids:\n')
        for c in centroids:
            fobj.write(' '.join([str(centroids[c][f]) for f in sorted(centroids[c])]) + '\n')
        fobj.write('\n')

        fobj.write('Clusters:\n')
        for c in clusters:
            fobj.write(' '.join([str(sent) for sent in sorted(clusters[c])]) + '\n')

        fobj.close()


class LM_interface(object):
    """abstract class; use either SRILM_interface or KENLM_interface"""

    def get_perplexity(self, text):
        cmd = [self.ppl_cmd] + self.ppl_options + self.options
        p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=open('/dev/null','w'))
        output = p.communicate(text)[0]

        scores = []

        # read sentence length and log-likelihood from SRILM output
        for k,line in enumerate(output.split('\n')):
            if k % 4 == 0 and line.startswith('file -:'):
                break
            elif k % 4 == 1:
                length = int(line.split()[2])
            elif k % 4 == 2:
                j = k // 4
                scores.append(-(float(line.split()[3]))/length)

        return scores


class SRILM_interface(LM_interface):
    """use SRILM for language model training / querying
    """

    def __init__(self, lm, order=1, text=None):
        self.training_cmd = 'ngram-count'
        self.ppl_cmd = 'ngram'

        self.training_options = ['-interpolate', '-kndiscount']
        self.ppl_options = ['-debug', '1', '-ppl', '-']
        self.options = ['-order', str(order), '-unk', '-lm', lm]
        self.name = lm

        if text and not os.path.exists(lm):
            self.train(text)


    def train(self, text):
        cmd = [self.training_cmd] + self.training_options + self.options + ['-text', text]
        sys.stderr.write('Training LM\n')
        sys.stderr.write(' '.join(cmd) + '\n')
        p = Popen(cmd)
        p.wait()


class KENLM_interface(LM_interface):
    """use Ken's tools for language model training / querying.
       ./ngram is a wrapper around query that emulates SRILM output
    """

    from config import LMPLZ_CMD

    def __init__(self, lm, order=1, text=None):
        self.training_cmd = LMPLZ_CMD
        self.ppl_cmd = './ngram'

        self.training_options = ['-S', '50%']
        self.ppl_options = ['-debug', '1', '-ppl', '-', '-lm', lm]
        self.options = ['-o', str(order)]
        self.name = lm

        if text and not os.path.exists(lm):
            self.train(text)


    def train(self, text):
        cmd = [self.training_cmd] + self.training_options + self.options
        sys.stderr.write('Training LM\n')
        sys.stderr.write(' '.join(cmd) + '\n')
        text = open(text,'r')
        lm = open(self.name,'w')
        p = Popen(cmd, stdin = text, stdout = lm)
        p.wait()
        text.close()
        lm.close()