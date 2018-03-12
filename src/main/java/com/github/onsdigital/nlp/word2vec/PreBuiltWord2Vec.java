package com.github.onsdigital.nlp.word2vec;

import com.github.onsdigital.utils.nlp.word2vec.Model;
import com.github.onsdigital.utils.nlp.word2vec.Word2VecHelper;
import org.deeplearning4j.models.word2vec.Word2Vec;

import java.io.IOException;

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

}
