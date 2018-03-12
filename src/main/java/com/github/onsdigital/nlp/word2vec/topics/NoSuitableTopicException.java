package com.github.onsdigital.nlp.word2vec.topics;

import com.github.onsdigital.utils.nlp.word2vec.Model;

/**
 * @author sullid (David Sullivan) on 07/03/2018
 * @project dp-word-utils
 */
public class NoSuitableTopicException extends Exception {

    private double[] vector;

    public NoSuitableTopicException(double[] vector, Model model) {
        super(String.format("Unable to locate suitable topic for model %s", model));
        this.vector = vector;
    }

    public double[] getVector() {
        return vector;
    }
}
