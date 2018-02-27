package com.github.onsdigital.nlp.word2vec;


import com.github.onsdigital.configuration.Configuration;
import com.github.onsdigital.utils.nlp.VectorModel;
import com.github.onsdigital.utils.nlp.Word2VecHelper;
import opennlp.tools.ngram.NGramGenerator;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.WhitespaceTokenizer;
import org.deeplearning4j.models.word2vec.Word2Vec;

import java.io.IOException;
import java.util.*;

/**
 * Simple utility class for computing lists of synonyms and their scores using word2vec
 * @author sullid (David Sullivan) on 21/02/2018
 * @project babbage
 */
public class SimilarityGenerator {

    private VectorModel model;
    private Word2Vec word2Vec;

    public SimilarityGenerator(VectorModel model) throws IOException {
        this.model = model;
        this.word2Vec = Word2VecHelper.getWord2Vec(model);
    }

    /**
     *
     * @param term
     * @return Map of Synonym -> score using the supplied word2vec model
     */
    private Map<String, Double> getSynonymsForTerm(String term, int numSearchTokens, boolean penaliseMoreTokens) {
        Map<String, Double> synonymMap = new HashMap<>();

        // Get the list of similar words
        Collection<String> similarWords = word2Vec.wordsNearest(term,
                Configuration.WORD_EMBEDDING.NUMBER_OF_SIMILAR_WORDS);

        for (String similar : similarWords) {
            // Tokenize
            String[] tokens = similar.split(Configuration.WORD_EMBEDDING.DELIMITER);

            // Compute ngrams from 1->Configuration.WORD_EMBEDDING.NUM_GRAM
            for (int n = 1; n <= Configuration.WORD_EMBEDDING.NUM_GRAMS; n++) {
                List<String> ngrams = NGramGenerator.generate(Arrays.asList(tokens), n, Configuration.WORD_EMBEDDING.DELIMITER);
                for (String ngram : ngrams) {
                    if (word2Vec.vocab().containsWord(ngram)) {
                        // If in vocabulary, add to the map

                        double score = word2Vec.similarity(term, ngram) * model.getWeight();
                        if (penaliseMoreTokens) {
                            // Penalise grams that contain more tokens than the original search term
                            String[] ngramTokens = ngram.split(Configuration.WORD_EMBEDDING.DELIMITER);
                            if (null != ngramTokens) {
                                int tokenDifference = Math.max(ngramTokens.length, numSearchTokens) - Math.min(ngramTokens.length, numSearchTokens);
                                score /= (double) (tokenDifference + 1);
                            }
                        }
                        synonymMap.put(ngram, score);
                    }
                }
            }
        }
        return synonymMap;
    }

    public Map<String, Map<String, Double>> getSimilaritiesForTerm(String searchTerm) {
        return this.getSimilaritiesForTerm(searchTerm, false);
    }

    /**
     *
     * @param searchTerm
     * @return Map of ngram strings to map of synonym -> similarity score
     */
    public Map<String, Map<String, Double>> getSimilaritiesForTerm(String searchTerm, boolean penaliseMoreTokens) {

        Map<String, Map<String, Double>> synonyms = new HashMap<>();

        // Initialise a simple whitespace tokenizer
        Tokenizer tokenizer = WhitespaceTokenizer.INSTANCE;

        // Get the tokens
        List<String> tokens = Arrays.asList(tokenizer.tokenize(searchTerm));

        // Compute ngrams from 1->Configuration.WORD_EMBEDDING.NUM_GRAM
        for (int n = 1; n <= Configuration.WORD_EMBEDDING.NUM_GRAMS; n++) {
            // n = 2 for bigrams
            List<String> grams = NGramGenerator.generate(tokens, n,
                    Configuration.WORD_EMBEDDING.DELIMITER);

            for (String gram : grams) {
                // Make sure the term is in our vocabulary
                if (word2Vec.vocab().containsWord(gram)) {
                    Map<String, Double> synonymMap = getSynonymsForTerm(gram, tokens.size(), penaliseMoreTokens);
                    synonyms.put(gram, synonymMap);
                }
            }
        }

        return synonyms;
    }

    public Map<String, Map<String, Double>> compoundSimilarities(Map<String, Map<String, Double>> similarWordsMapping) {
        // Compute compound similarities
        Map<String, Map<String, Double>> compoundSynonyms = new HashMap<>();
        for (String key : similarWordsMapping.keySet()) {
            Map<String, Double> scores = similarWordsMapping.get(key);
            Map<String, Double> compoundScores = new HashMap<>();

            for (String term : scores.keySet()) {
                double score = scores.get(term);
                double count = 0.0d;
                for (String otherTerm : scores.keySet()) {
                    if (scores.get(otherTerm) > Configuration.WORD_EMBEDDING.CONFIDENCE_THRESHOLD) {
                        if (!otherTerm.equals(term) && word2Vec.vocab().containsWord(term) &&
                                word2Vec.vocab().containsWord(otherTerm)) {
                            score += word2Vec.similarity(term, otherTerm);
                        }
                        count++;
                    }
                }
                double newScore = score/count;
                compoundScores.put(term, newScore);
            }
            compoundSynonyms.put(key, compoundScores);
        }
        return compoundSynonyms;
    }
}
