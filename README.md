multidomain_smt
================

A project of the Computational Linguistics Group at the University of Zurich (http://www.cl.uzh.ch).

Project Homepage: http://github.com/rsennrich/multidomain_smt

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation


ABOUT
-----


This repository is a sample implementation of the clustering method described in:

    Rico Sennrich, Holger Schwenk and Walid Aransa. 2013. A Multi-Domain Translation Model Framework for Statistical Machine Translation. In Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics (ACL 2013), p. 382-840.


REQUIREMENTS
------------

The program requires Python (2.6 or greater), GIZA++ and the Moses toolkit (compiled with XML-RCP-C and DLIB). Set the paths in `config.py`.


USAGE
-----

A number of options have to be set in `config.py`:

    - paths to Moses binaries and GIZA++

    - source-side language models (or parallel texts) for clustering: LM_TEXTS
    - a parallel development set to be clustered: DEV_L1/DEV_L2
    - K, the number of clusters in K-means clustering

    - a test set TEST_SET

Also, the translation models to be combined need to be pre-trained, converted into the right format with `/path/to/moses/scripts/training/create_count_tables.py`, and referenced in MOSES_CFG.
See `demo/moses.ini` for an example config file, and http://www.statmt.org/moses/?n=Moses.AdvancedFeatures for a documentation of the MultiModelCounts phrase table type.

Executing the program:

    python main.py

will do the following:

    - cluster the development set into K clusters using source side language models
    - extract a set of phrase pairs for each cluster (using GIZA++ for word alignment and heuristic phrase extraction)
    - for each cluster, optimize the instance weights of the component models in demo/moses.ini
    - translate the test set. for each sentence:
        - assign it to the closest cluster
        - translate the sentence using the optimized instance weights that correspond to this cluster

the script saves the clustering information and instance weights to a file (`persistent_data.txt` and `persistent_weights.txt`) so that you can repeat the translation step with new texts.


CONTACT
-------

For questions and feeback, please contact sennrich@cl.uzh.ch or use the GitHub repository.
