package com.github.onsdigital.nlp.word2vec;

import com.github.onsdigital.utils.nlp.word2vec.Model;

/**
 * @author sullid (David Sullivan) on 07/03/2018
 * @project dp-word-utils
 */
public class NotInVocabularyException extends Exception {
    public NotInVocabularyException(String word, Model model) {
        super(String.format("Word '%s' not in vocabulary for model %s", word, model));
    }
}
