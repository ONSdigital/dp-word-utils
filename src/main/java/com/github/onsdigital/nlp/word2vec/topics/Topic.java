package com.github.onsdigital.nlp.word2vec.topics;

import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collection;
import java.util.List;

/**
 * @author sullid (David Sullivan) on 07/03/2018
 * @project dp-word-utils
 */
public class Topic {

    private final int clusterNumber;
    private final double[] topicVector;
    private final Word2Vec word2Vec;

    public Topic(int clusterNumber, double[] topicVector, final Word2Vec word2Vec) {
        this.clusterNumber = clusterNumber;
        this.topicVector = topicVector;
        this.word2Vec = word2Vec;
    }

    public int getClusterNumber() {
        return clusterNumber;
    }

    public String nearestWord() {
        Collection<String> nearestWord = this.nearestWords(this.getTopicINDArray(), 1);
        return nearestWord.iterator().next();
    }

    public Collection<String> nearestWords(INDArray vector, int number) {
        return this.word2Vec.wordsNearest(vector, number);
    }

    public double[] getTopicVector() {
        return this.topicVector;
    }

    public INDArray getTopicINDArray() {
        return Nd4j.create(this.getTopicVector());
    }
}
