package com.github.onsdigital.utils.nlp.word2vec;

import com.github.onsdigital.utils.GZipFile;
import org.apache.commons.compress.compressors.gzip.GzipUtils;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.reader.ModelUtils;
import org.deeplearning4j.models.word2vec.Word2Vec;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Utility class to provide cached loading of Word2vec models.
 * @author sullid (David Sullivan) on 27/02/2018
 * @project dp-word-utils
 */
public class Word2VecHelper {

    private static Map<Model, Word2Vec> modelMap = new ConcurrentHashMap<>();

    private static Word2Vec loadModel(Model model) throws IOException {
        // Loads a word2vec model using a *.vec model on disk
        ClassLoader classLoader = Word2VecHelper.class.getClassLoader();
        URL fileUrl = classLoader.getResource(model.getFilename());

        if (null == fileUrl) {
            throw new IOException(String.format("Unable to locate model file for model %s with filename: %s", model,
                    model.getFilename()));
        }

        File modelFile;
        if (GzipUtils.isCompressedFilename(fileUrl.getFile())) {
            GZipFile gZipFile = new GZipFile(fileUrl.getFile());
            modelFile = gZipFile.gunzip();
        } else {
            modelFile = new File(fileUrl.getFile());
        }

        Word2Vec word2vec = WordVectorSerializer.readWord2VecModel(modelFile);
        return word2vec;
    }

    public static void init(Model model) throws Exception {
        if (!modelMap.containsKey(model)) {
            Word2Vec word2Vec = loadModel(model);

            modelMap.put(model, word2Vec);
        } else {
            throw new Exception(String.format("Model %s already initialised!", model));
        }
    }

    public static Word2Vec setModelUtils(Model model, ModelUtils modelUtils) {
        modelMap.get(model).setModelUtils(modelUtils);
        return modelMap.get(model);
    }

    public static Word2Vec getWord2Vec(Model model) throws IOException {
        if (!modelMap.containsKey(model)) {
            modelMap.put(model, loadModel(model));
        }
        return modelMap.get(model);
    }

    public enum ONSModel implements Model {
        ONS_FT("ons_ft.vec.gz");

        private String filename;
        private float weight = 1.0f;

        ONSModel(String filename) {
            this.filename = filename;
        }

        ONSModel(String filename, float weight) {
            this.filename = filename;
            this.weight = weight;
        }

        public String getModelFilename() {
            return this.filename;
        }

        public float getWeight() {
            return weight;
        }
    }

}
