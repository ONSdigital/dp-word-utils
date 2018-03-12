package com.github.onsdigital.nlp.word2vec.topics;

import org.apache.commons.math3.ml.clustering.Clusterable;

/**
 * @author sullid (David Sullivan) on 07/03/2018
 * @project dp-word-utils
 */
public class Word2VecClusterable implements Clusterable {

    private final String word;
    private final double[] vector;

    protected Word2VecClusterable(String word, double[] vector) {
        this.word = word;
        this.vector = vector;
    }

    @Override
    public double[] getPoint() {
        return this.vector;
    }
}
