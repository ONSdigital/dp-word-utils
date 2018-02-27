package com.github.onsdigital.configuration;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author sullid (David Sullivan) on 27/02/2018
 * @project dp-word-utils
 */
public class Configuration {

    private static final Logger LOGGER = LoggerFactory.getLogger(Configuration.class);

    private static String getProperty(String key) {
        return System.getenv(key);
    }

    private static String defaultIfNotExists(String key, String defaultValue) {
        String value = getProperty(key);
        return value == null ? defaultValue : value;
    }

    private static Float defaultFloatIfNotExists(String key, Float defaultValue) {
        try {
            String property = getProperty(key);
            return property == null ? defaultValue : Float.parseFloat(getProperty(key));
        } catch (Exception e) {
            LOGGER.error(String.format("Unable to parse property %s=%s to int.", key, getProperty(key)));
            return defaultValue;
        }
    }

    public static class WORD_EMBEDDING {

        private static final String WORD2VEC_DIRECTORY_KEY = "WORD2VEC_DIRECTORY";
        private static final String NUMBER_OF_SIMILAR_WORDS_KEY = "NUMBER_OF_SIMILAR_WORDS";
        private static final String DELIMITER_KEY = "WORD2VEC_DELIMITER";
        private static final String NUM_GRAMS_KEY = "NGRAMS_SIMILARITY";
        private static final String CONFIDENCE_THRESHOLD_KEY = "CONFIDENCE_THRESHOLD";

        public static final String WORD2VEC_DIRECTORY;
        public static final Integer NUMBER_OF_SIMILAR_WORDS;
        public static final String DELIMITER;
        public static final Integer NUM_GRAMS;
        public static final Float CONFIDENCE_THRESHOLD;

        static {
            WORD2VEC_DIRECTORY = defaultIfNotExists(WORD2VEC_DIRECTORY_KEY, "word2vec");
            LOGGER.info(String.format("Using word2vec directory: %s", WORD2VEC_DIRECTORY));

            NUMBER_OF_SIMILAR_WORDS = defaultFloatIfNotExists(NUMBER_OF_SIMILAR_WORDS_KEY, 10.0f).intValue();
            LOGGER.info(String.format("NUMBER_OF_SIMILAR_WORDS = %d", NUMBER_OF_SIMILAR_WORDS));

            DELIMITER = defaultIfNotExists(DELIMITER_KEY, "_");
            LOGGER.info(String.format("Using delimiter: %s", DELIMITER));

            NUM_GRAMS = defaultFloatIfNotExists(NUM_GRAMS_KEY, 3.0f).intValue();
            LOGGER.info(String.format("Using %d grams", NUM_GRAMS));

            CONFIDENCE_THRESHOLD = defaultFloatIfNotExists(CONFIDENCE_THRESHOLD_KEY, 0.3f);
            LOGGER.info(String.format("Confidence threshold is %f", CONFIDENCE_THRESHOLD));
        }

    }

}
