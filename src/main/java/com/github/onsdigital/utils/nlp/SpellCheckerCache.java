package com.github.onsdigital.utils.nlp;

import com.github.onsdigital.nlp.word2vec.SpellChecker;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * @author sullid (David Sullivan) on 27/02/2018
 * @project dp-word-utils
 */
public class SpellCheckerCache {

    private static Map<VectorModel, SpellChecker> spellCheckerMap = new ConcurrentHashMap<>();

    public static SpellChecker forModel(VectorModel model) throws IOException {
        if (!spellCheckerMap.containsKey(model)) {
            SpellChecker spellChecker = new SpellChecker(model);
            spellCheckerMap.put(model, spellChecker);
        }
        return spellCheckerMap.get(model);
    }

}
