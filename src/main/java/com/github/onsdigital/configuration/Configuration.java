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
        private static final String DELIMITER_KEY = "WORD2VEC_DELIMITER";

        public static final String WORD2VEC_DIRECTORY;
        public static final String DELIMITER;

        static {
            WORD2VEC_DIRECTORY = defaultIfNotExists(WORD2VEC_DIRECTORY_KEY, "word2vec");
            LOGGER.info(String.format("Using word2vec directory: %s", WORD2VEC_DIRECTORY));

            DELIMITER = defaultIfNotExists(DELIMITER_KEY, "_");
            LOGGER.info(String.format("Using delimiter: %s", DELIMITER));
        }

    }

}
