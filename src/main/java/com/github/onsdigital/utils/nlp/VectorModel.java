package com.github.onsdigital.utils.nlp;

import com.github.onsdigital.configuration.Configuration;

import java.nio.file.Paths;

/**
 * @author sullid (David Sullivan) on 27/02/2018
 * @project dp-word-utils
 */
public interface VectorModel {

    default String getFilename() {
        return Paths.get(Configuration.WORD_EMBEDDING.WORD2VEC_DIRECTORY,
                getModelFilename()).toString();
    }

    String getModelFilename();

    double getWeight();

}
