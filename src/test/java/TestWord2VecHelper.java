import com.github.onsdigital.utils.nlp.VectorModel;
import com.github.onsdigital.utils.nlp.Word2VecHelper;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.junit.Assert;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * @author sullid (David Sullivan) on 27/02/2018
 * @project dp-word-utils
 */
public class TestWord2VecHelper {

    private static final Logger LOGGER = LoggerFactory.getLogger(TestWord2VecHelper.class);

    private static final VectorModel model = Word2VecHelper.ONSModel.ONS_FT;

    @Test
    public void testModelLoad() {
        try {
            Word2Vec word2Vec = Word2VecHelper.getWord2Vec(model);

            Assert.assertNotNull(word2Vec);
        } catch (IOException e) {
            LOGGER.error(String.format("Error loading model %s", model), e);
            Assert.fail(e.getMessage());
        }
    }

}
