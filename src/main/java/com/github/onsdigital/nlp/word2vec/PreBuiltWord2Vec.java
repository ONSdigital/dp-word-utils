package com.github.onsdigital.nlp.word2vec;

import com.github.onsdigital.utils.nlp.word2vec.Model;
import com.github.onsdigital.utils.nlp.word2vec.Word2VecHelper;
import org.deeplearning4j.models.word2vec.Word2Vec;

import java.io.IOException;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

/**
 * @author sullid (David Sullivan) on 07/03/2018
 * @project dp-word-utils
 */
public class PreBuiltWord2Vec {

    protected final Model model;
    protected final Word2Vec word2Vec;

    public PreBuiltWord2Vec(Model model) throws IOException {
        this.model = model;
        this.word2Vec = Word2VecHelper.getWord2Vec(model);
    }

    public Model getModel() {
        return model;
    }

    public Word2Vec getWord2Vec() {
        return word2Vec;
    }

    /**
     *
     * @param word
     * @return true if word is in model vocabulary, false otherwise
     */
    public boolean inVocabulary(String word) {
        return this.word2Vec.hasWord(word);
    }

    /**
     *
     * @param word
     * @param number
     * @return Map of similar words to their score (cosine distance).
     */
    public Map<String, Double> similarTerms(String word, int number) {
        Map<String, Double> similarTermsMapping = new HashMap<>();

        Collection<String> similarTerms = this.word2Vec.wordsNearest(word, number);
        for (String similarTerm : similarTerms) {
            similarTermsMapping.put(similarTerm, this.word2Vec.similarity(word, similarTerm));
        }

        return similarTermsMapping;
    }
}
