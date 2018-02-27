/**
 * Based on the Java 8 Spelling Corrector by Peter Kuhar, with minor changes.
 * Copyright 2016 Peter Kuhar.
 *
 * Open source code under MIT license: http://www.opensource.org/licenses/mit-license.php
 */

package com.github.onsdigital.nlp.word2vec;

import com.github.onsdigital.utils.nlp.VectorModel;
import com.github.onsdigital.utils.nlp.Word2VecHelper;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;

import java.io.IOException;
import java.util.*;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * A simple, lightweight, spell checker which uses word2vec models for it's dictionary.
 * @author sullid (David Sullivan) on 27/02/2018
 * @project dp-word-utils
 */
public class SpellChecker {

    private Map<String, Integer> dictionary = new HashMap<>();
    private final VectorModel vectorModel;
    private final Word2Vec word2Vec;

    public SpellChecker(VectorModel vectorModel) throws IOException {
        // Load the word2vec model
        this.vectorModel = vectorModel;
        this.word2Vec = Word2VecHelper.getWord2Vec(this.vectorModel);

        // Build the dictionary
        VocabCache<VocabWord> vocabCache = this.word2Vec.vocab();
        Iterator<VocabWord> it = vocabCache.tokens().iterator();
        while (it.hasNext()) {
            VocabWord vocabWord = it.next();
            this.dictionary.put(vocabWord.getWord(), vocabWord.getIndex());
        }
    }

    private final Stream<String> edits1(final String word){
        Stream<String> deletes    = IntStream.range(0, word.length())  .mapToObj((i) -> word.substring(0, i) + word.substring(i + 1));
        Stream<String> replaces   = IntStream.range(0, word.length())  .mapToObj((i)->i).flatMap( (i) -> "abcdefghijklmnopqrstuvwxyz".chars().mapToObj( (c) ->  word.substring(0,i) + (char)c + word.substring(i+1) )  );
        Stream<String> inserts    = IntStream.range(0, word.length()+1).mapToObj((i)->i).flatMap( (i) -> "abcdefghijklmnopqrstuvwxyz".chars().mapToObj( (c) ->  word.substring(0,i) + (char)c + word.substring(i) )  );
        Stream<String> transposes = IntStream.range(0, word.length()-1).mapToObj((i)-> word.substring(0,i) + word.substring(i+1,i+2) + word.charAt(i) + word.substring(i+2) );
        return Stream.of( deletes,replaces,inserts,transposes ).flatMap((x)->x);
    }

    private final Stream<String> known(Stream<String> words){
        return words.filter( (word) -> this.dictionary.containsKey(word) );
    }

    public final Integer getRank(String key) {
        // Use inverse of rank as proxy of probability.
        // Returns '0' if word not in vocabulary
        return this.dictionary.containsKey(key) ? -1 * this.dictionary.get(key) : 0;
    }

    public final String correct(String word){
        Optional<String> e1 = known(edits1(word)).max(Comparator.comparingInt(a -> this.getRank(a)));
        if(e1.isPresent()) {
            return this.dictionary.containsKey(word) ? word : e1.get();
        }
        Optional<String> e2 = known(edits1(word).map(this::edits1).flatMap((x)->x)).max(Comparator.comparingInt(a -> this.getRank(a)));
        return (e2.orElse(word));
    }

    public final boolean inVocabulary(String key) {
        return this.dictionary.containsKey(key);
    }

}
