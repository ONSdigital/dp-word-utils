import com.github.onsdigital.nlp.word2vec.SpellChecker;
import com.github.onsdigital.utils.nlp.VectorModel;
import com.github.onsdigital.utils.nlp.Word2VecHelper;
import org.junit.Assert;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * @author sullid (David Sullivan) on 27/02/2018
 * @project dp-word-utils
 */
public class TestSpellChecker {

    private static final Logger LOGGER = LoggerFactory.getLogger(TestSpellChecker.class);

    private static final VectorModel model = Word2VecHelper.ONSModel.ONS_FT;

    private static final List<String> testWords;
    private static final List<String> correctWords;

    static {
        testWords = new ArrayList<String>() {{
            add("cli");
            add("murdr");
            add("rpo");
            add("crme");
        }};

        correctWords = new ArrayList<String>() {{
            add("cpi");
            add("murder");
            add("rpi");
            add("crime");
        }};
    }

    @Test
    public void testSpellCheck() {
        try {
            SpellChecker spellChecker = new SpellChecker(model);

            for (int i = 0; i < testWords.size(); i++) {
                String testWord = testWords.get(i);
                String correctWord = correctWords.get(i);

                String spellChecked = spellChecker.correct(testWord);
                assertEquals(correctWord, spellChecked);
            }
        } catch (IOException e) {
            LOGGER.error(String.format("Error initialising SpellChecker with model %s", model), e);
            Assert.fail(e.getMessage());
        }
    }

}
