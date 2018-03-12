package com.github.onsdigital.nlp.word2vec.topics.distance;

import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * @author sullid (David Sullivan) on 07/03/2018
 * @project dp-word-utils
 */
public class CosineDistance implements DistanceMeasure {
    @Override
    public double compute(double[] doubles, double[] doubles1) {
        return Transforms.cosineDistance(Nd4j.create(doubles), Nd4j.create(doubles1));
    }
}