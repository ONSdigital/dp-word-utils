package com.github.onsdigital.utils.nlp.word2vec;

import com.github.onsdigital.nlp.word2vec.PreBuiltWord2Vec;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * @author sullid (David Sullivan) on 12/03/2018
 * @project dp-word-utils
 */
public class PreBuiltModels {

    private static Map<Model, PreBuiltWord2Vec> preBuiltWord2VecMap = new ConcurrentHashMap<>();

    public static PreBuiltWord2Vec getModel(Model model) throws IOException {
        if (!preBuiltWord2VecMap.containsKey(model)) {
            preBuiltWord2VecMap.put(model, new PreBuiltWord2Vec(model));
        }
        return preBuiltWord2VecMap.get(model);
    }

    public static PreBuiltWord2Vec remove(Model model) {
        return preBuiltWord2VecMap.remove(model);
    }

}
