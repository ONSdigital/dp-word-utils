package com.github.onsdigital.nlp.word2vec.topics;

import com.github.onsdigital.utils.TimeUtils;
import com.github.onsdigital.utils.nlp.word2vec.Model;
import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import smile.clustering.DBScan;
import smile.math.distance.Distance;

import java.io.IOException;
import java.util.*;

/**
 * @author sullid (David Sullivan) on 07/03/2018
 * @project dp-word-utils
 */
public class DBScanTopicModel extends TopicModel {

    private static final Logger LOGGER = LoggerFactory.getLogger(DBScanTopicModel.class);

    private DistanceWrapper distanceMeasure;

    private DBScan<Word2VecClusterable> dbScan;

    public DBScanTopicModel(Model model, DistanceMeasure distanceMeasure) throws IOException {
        this(model, distanceMeasure, null);
    }

    public DBScanTopicModel(Model model, DistanceMeasure distanceMeasure, Set<String> words) throws IOException {
        super(model, words);
        this.distanceMeasure = new DistanceWrapper(distanceMeasure);
    }

    @Override
    public double distance(double[] vector1, double[] vector2) {
        return this.distanceMeasure.getDistanceMeasure().compute(vector1, vector2);
    }

    @Override
    public List<Topic> buildTopicList() {
        if (null == super.topics) {
            LOGGER.info("Initialising DBScan...");

            List<Word2VecClusterable> clusterables = new ArrayList<>();
            words.stream()
                    .filter(word -> super.word2Vec.hasWord(word))
                    .forEach(word -> clusterables.add(new Word2VecClusterable(word, super.word2Vec.getWordVector(word))));

            LOGGER.info(String.format("Running dbscan with %d words...", clusterables.size()));
            long start = System.currentTimeMillis();
            this.dbScan = new DBScan<>(clusterables.toArray(new Word2VecClusterable[clusterables.size()]),
                    this.distanceMeasure, 5, 0.5);
            long stop = System.currentTimeMillis();
            long duration = (stop - start);
            LOGGER.info(String.format("Clustered %d words in %s", clusterables.size(), TimeUtils.millisToShortDHMS(duration)));

            // Init the topic list
            super.topics = new ArrayList<>();
            // Populate
            // Compute cluster centres
            LOGGER.info("Computing cluster centres...");
            Map<Integer, Double[]> posMap = new HashMap<>();
            Map<Integer, Integer> countMap = new HashMap<>();
            for (Word2VecClusterable clusterable : clusterables) {
                int cluster = this.dbScan.predict(clusterable);
                double[] vec = clusterable.getPoint();
                if (posMap.containsKey(cluster)) {
                    Double[] pos = posMap.get(cluster);
                    for (int j = 0; j < pos.length; j++){
                        pos[j] += vec[j];
                    }
                    countMap.replace(cluster, countMap.get(cluster) + 1);
                } else {
                    Double[] init = new Double[vec.length];
                    for (int j = 0; j < init.length; j++) {
                        init[j] = vec[j];
                    }
                    posMap.put(cluster, init);
                    countMap.put(cluster, 1);
                }
            }

            // Average positions and create topics
            for (Integer cluster : posMap.keySet()) {
                Double[] clusterCentre = posMap.get(cluster);
                double[] centre = new double[clusterCentre.length];
                for (int j = 0; j < clusterCentre.length; j++) {
                    centre[j] = clusterCentre[j] / (double) countMap.get(cluster);
                }
                super.topics.add(new Topic(cluster, centre, super.word2Vec));
            }
        duration = System.currentTimeMillis() - stop;
        LOGGER.info(String.format("Computed centres in %s", TimeUtils.millisToShortDHMS(duration)));
        }
        return super.topics;
    }

    private static class DistanceWrapper implements Distance<Word2VecClusterable> {

        private DistanceMeasure distanceMeasure;

        public DistanceWrapper(DistanceMeasure distanceMeasure) {
            this.distanceMeasure = distanceMeasure;
        }

        @Override
        public double d(Word2VecClusterable word2VecClusterable, Word2VecClusterable t1) {
            return this.distanceMeasure.compute(word2VecClusterable.getPoint(), t1.getPoint());
        }

        public DistanceMeasure getDistanceMeasure() {
            return distanceMeasure;
        }
    }
}
