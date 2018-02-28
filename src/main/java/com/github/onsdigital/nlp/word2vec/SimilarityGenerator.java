package com.github.onsdigital.nlp.word2vec;


import com.github.onsdigital.configuration.Configuration;
import com.github.onsdigital.utils.nlp.VectorModel;
import com.github.onsdigital.utils.nlp.Word2VecHelper;
import org.deeplearning4j.models.word2vec.Word2Vec;

import java.io.IOException;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

/**
 * Simple utility class for computing lists of synonyms and their scores using word2vec
 * @author sullid (David Sullivan) on 21/02/2018
 * @project dp-word-utils
 */
public class SimilarityGenerator {

    private VectorModel model;
    private Word2Vec word2Vec;

    public SimilarityGenerator(VectorModel model) throws IOException {
        this.model = model;
        this.word2Vec = Word2VecHelper.getWord2Vec(model);
    }

    public boolean inVocabulary(String word) {
        return this.word2Vec.vocab().containsWord(word);
    }

    public Map<String, Double> similarTerms (String term, int numSimilarTerms) {
        return this.similarTerms(term, numSimilarTerms, Configuration.WORD_EMBEDDING.DELIMITER);
    }

    public Map<String, Double> similarTerms (String term, int numSimilarTerms, String delimiter) {
        // Initialise the map of term -> similarity score
        Map<String, Double> similarityMap = new HashMap<>();

        // Get the similar terms
        Collection<String> similarTerms = this.word2Vec.wordsNearest(term, numSimilarTerms);

        // Populate map with scores
        for (String similarTerm : similarTerms) {
            similarityMap.put(similarTerm, this.word2Vec.similarity(term, similarTerm));
        }

        return similarityMap;
    }
}
