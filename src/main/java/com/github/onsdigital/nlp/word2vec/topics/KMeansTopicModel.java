package com.github.onsdigital.nlp.word2vec.topics;

import com.github.onsdigital.nlp.word2vec.NotInVocabularyException;
import com.github.onsdigital.nlp.word2vec.topics.distance.CosineDistance;
import com.github.onsdigital.utils.TimeUtils;
import com.github.onsdigital.utils.nlp.word2vec.Model;
import com.github.onsdigital.utils.nlp.word2vec.Word2VecHelper;
import com.google.common.collect.Sets;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.log4j.BasicConfigurator;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * @author sullid (David Sullivan) on 07/03/2018
 * @project dp-word-utils
 */
public class KMeansTopicModel extends TopicModel {

    private static final Logger LOGGER = LoggerFactory.getLogger(KMeansTopicModel.class);

    public static final int DEFAULT_MAX_ITERATIONS = 1000;
    public static final int DEFAULT_SEED = 12345;

    private final int numClusters;
    private final int maxIterations;
    private final int seed;
    private final DistanceMeasure distanceMeasure;

    private KMeansPlusPlusClusterer<Word2VecClusterable> kMeans;

    public KMeansTopicModel(Model model, int numClusters) throws IOException {
        this(model, numClusters, new CosineDistance());
    }

    public KMeansTopicModel(Model model, int numClusters, DistanceMeasure distanceMeasure) throws IOException {
        this(model, numClusters, DEFAULT_MAX_ITERATIONS, distanceMeasure, DEFAULT_SEED, null);
    }

    public KMeansTopicModel(Model model, int numClusters, int maxIterations, DistanceMeasure distanceMeasure, int seed, Set<String> words) throws IOException {
        super(model, words);
        this.numClusters = numClusters;
        this.maxIterations = maxIterations;
        this.seed = seed;
        this.distanceMeasure = distanceMeasure;

        // Init the KMeans clustering pipeline
        this.kMeans = withSeed(this.numClusters, this.maxIterations, this.distanceMeasure, this.seed);
    }

    @Override
    public double distance(double[] vector1, double[] vector2) {
        return this.distanceMeasure.compute(vector1, vector2);
    }

    @Override
    public List<Topic> buildTopicList() {
        if (null == super.topics) {
            LOGGER.info("Initialising KMeans...");
            // Cluster words into topics
            List<Word2VecClusterable> clusterables = new ArrayList<>();
            this.words.stream()
                    .forEach(word -> {
                        try {
                            clusterables.add(this.getClusterableForWord(word));
                        } catch (NotInVocabularyException e) {
                            LOGGER.warn(String.format("Word '%s' not in vocabulary. Ignoring.", word));
                        }
                    });

            LOGGER.info(String.format("Clustering %d words...", clusterables.size()));
            long start = System.currentTimeMillis();
            List<CentroidCluster<Word2VecClusterable>> centroidClusters = this.kMeans.cluster(clusterables);
            long stop = System.currentTimeMillis();
            long duration = (stop - start);

            // Init the topic list
            super.topics = new ArrayList<>();
            // Populate
            for (int i = 0; i < centroidClusters.size(); i++) {
                this.topics.add(new Topic(i, centroidClusters.get(i).getCenter().getPoint(), super.word2Vec));
            }
            LOGGER.info(String.format("Clustered %d words in %s", clusterables.size(), TimeUtils.millisToShortDHMS(duration)));
        }
        return this.topics;
    }

    private Word2VecClusterable getClusterableForWord(String word) throws NotInVocabularyException {
        if (!super.word2Vec.hasWord(word)) {
            throw new NotInVocabularyException(word, super.model);
        }
        double[] vector = super.word2Vec.getWordVector(word);
        return new Word2VecClusterable(word, vector);
    }

    private static KMeansPlusPlusClusterer<Word2VecClusterable> withSeed(int numClusters, int maxIterations, DistanceMeasure distanceMeasure, int seed) {
        RandomGenerator randomGenerator = new JDKRandomGenerator();
        randomGenerator.setSeed(seed);

        return new KMeansPlusPlusClusterer<>(numClusters, maxIterations, distanceMeasure, randomGenerator);
    }

}
