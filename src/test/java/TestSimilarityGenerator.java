import com.github.onsdigital.nlp.word2vec.SimilarityGenerator;
import com.github.onsdigital.utils.nlp.VectorModel;
import com.github.onsdigital.utils.nlp.Word2VecHelper;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author sullid (David Sullivan) on 27/02/2018
 * @project dp-word-utils
 */
public class TestSimilarityGenerator {

    private static final Logger LOGGER = LoggerFactory.getLogger(TestSimilarityGenerator.class);

    private static final VectorModel model = Word2VecHelper.ONSModel.ONS_FT;

    private static final String term = "cpi";

    private static final Map<String, Map<String, Double>> expected;

    static {
        // Sub-set of top terms expected. Note, we don't include more terms as the default configuration
        // could have been changed.
        expected = new HashMap<String, Map<String, Double>>() {{
            put("cpi", new HashMap<String, Double>() {{
                put("consumer_price", 0.6109464168548584d);
                put("cpih", 0.6257569193840027d);
                put("cpi", 1.0d);
            }});
        }};
    }

    @Test
    public void testSimilarity() {
        try {
            SimilarityGenerator similarityGenerator = new SimilarityGenerator(model);

            Map<String, Map<String, Double>> similarTerms = similarityGenerator.getSimilaritiesForTerm(term);

            for (String key : expected.keySet()) {
                assertTrue(similarTerms.containsKey(key));

                Map<String, Double> expectedScores = expected.get(key);
                for (String expectedKey : expectedScores.keySet()) {
                    assertTrue(similarTerms.get(key).containsKey(expectedKey));
                    Double score = similarTerms.get(key).get(expectedKey);
                    assertEquals(expectedScores.get(expectedKey), score);
                }
            }
        } catch (IOException e) {
            LOGGER.error(String.format("Error initialising SimilarityGenerator with model %s", model), e);
            Assert.fail(e.getMessage());
        }
    }

}
