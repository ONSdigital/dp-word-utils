package com.github.onsdigital.utils.nlp.word2vec;

import com.github.onsdigital.nlp.word2vec.SpellChecker;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * @author sullid (David Sullivan) on 12/03/2018
 * @project dp-word-utils
 */
public class SpellCheckers {

    private static Map<Model, SpellChecker> spellCheckers = new ConcurrentHashMap<>();

    public static SpellChecker getSpellChecker(Model model) throws IOException {
        if (!spellCheckers.containsKey(model)) {
            spellCheckers.put(model, new SpellChecker(model));
        }
        return spellCheckers.get(model);
    }

    public static SpellChecker remove(Model model) {
        return spellCheckers.remove(model);
    }

}
