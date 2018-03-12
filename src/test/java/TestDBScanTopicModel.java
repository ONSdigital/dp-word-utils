import com.github.onsdigital.nlp.word2vec.topics.DBScanTopicModel;
import com.github.onsdigital.nlp.word2vec.topics.Topic;
import com.github.onsdigital.nlp.word2vec.topics.distance.CosineDistance;
import com.github.onsdigital.utils.nlp.word2vec.Word2VecHelper;
import com.google.common.collect.Sets;
import org.apache.log4j.BasicConfigurator;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.junit.Assert;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static com.github.onsdigital.nlp.word2vec.topics.TopicModel.getRandomSubList;
import static org.junit.Assert.assertTrue;

/**
 * @author sullid (David Sullivan) on 12/03/2018
 * @project dp-word-utils
 */
public class TestDBScanTopicModel {

    private static final Logger LOGGER = LoggerFactory.getLogger(TestDBScanTopicModel.class);

    @Test
    public void test() {
        BasicConfigurator.configure();

        Word2Vec word2Vec = null;
        try {
            word2Vec = Word2VecHelper.getWord2Vec(Word2VecHelper.ONSModel.ONS_FT);
        } catch (IOException e) {
            LOGGER.error("Error initialising model", e);
            Assert.fail(e.getMessage());
        }
        List<String> words = new ArrayList<>(word2Vec.vocab().words());

        LOGGER.info(String.format("Got %d words", words.size()));
        List<String> subset = getRandomSubList(words, words.size() / 100);

        DBScanTopicModel model = null;
        try {
            model = new DBScanTopicModel(Word2VecHelper.ONSModel.ONS_FT, new CosineDistance(), Sets.newHashSet(subset));
        } catch (IOException e) {
            LOGGER.error("Error initialising model", e);
            Assert.fail(e.getMessage());
        }
        List<Topic> topics = model.buildTopicList();
        assertTrue(topics.size() > 0);  // with DBScan, we do not know number of clusters beforehand
    }

}
