package com.github.onsdigital.nlp.word2vec.topics;

import com.github.onsdigital.nlp.word2vec.PreBuiltWord2Vec;
import com.github.onsdigital.utils.nlp.word2vec.Model;
import com.google.common.collect.Sets;

import java.io.IOException;
import java.util.List;
import java.util.Random;
import java.util.Set;

/**
 * @author sullid (David Sullivan) on 07/03/2018
 * @project dp-word-utils
 */
public abstract class TopicModel extends PreBuiltWord2Vec {

    // To be populated
    protected List<Topic> topics;
    protected Set<String> words;

    public TopicModel(Model model) throws IOException {
        this(model, null);
    }

    public TopicModel(Model model, Set<String> words) throws IOException {
        super(model);

        if (words == null) {
            words = Sets.newHashSet(super.word2Vec.vocab().words());
        }
        this.words = words;
    }

    public abstract double distance(double[] vector1, double[] vector2);

    public abstract List<Topic> buildTopicList();

    public Topic topicForVector(double[] vector) throws NoSuitableTopicException {
        // Returns the topic closest to this vector
        Topic nearestTopic = null;
        double distance = Double.MAX_VALUE;
        for (Topic topic : this.buildTopicList()) {
            double distanceToTopic = this.distance(topic.getTopicVector(), vector);
            if (distanceToTopic < distance) {
                nearestTopic = topic;
                distance = distanceToTopic;
            }
        }
        if (null == nearestTopic) {
            throw new NoSuitableTopicException(vector, super.model);
        }
        return nearestTopic;
    }

    public static <T> List<T> getRandomSubList(List<T> input, int subsetSize) {
        Random r = new Random();
        int inputSize = input.size();
        for (int i = 0; i < subsetSize; i++)
        {
            int indexToSwap = i + r.nextInt(inputSize - i);
            T temp = input.get(i);
            input.set(i, input.get(indexToSwap));
            input.set(indexToSwap, temp);
        }
        return input.subList(0, subsetSize);
    }

}
