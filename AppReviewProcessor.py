import os
import re
import shlex
import subprocess
from collections import OrderedDict

import nltk
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer

from FileHandler import PickleHandler


class AppReviewProcessor:
    def __init__(self):
        # needed for the full feature set
        self.vectorizer_bow = self.get_bow_vectorizer()
        self.vectorizer_bigram = self.get_bigram_vectorizer()

    def process_many(self, app_reviews):
        processed_app_reviews = list()
        for app_review in app_reviews:
            processed_app_reviews.append(self.process(app_review))

        return processed_app_reviews

    def process(self, app_review):
        original_review = app_review["title"] + " " + app_review["body"]
        review = original_review.strip().lower()
        review = NLPHelper.remove_stopwords(review)
        review = NLPHelper.lem(review)
        review_tense = NLPHelper.count_tense_occurrences(review)
        review_sentiment = NLPHelper.get_sentiment(original_review)

        processed_app_review = app_review
        if "_id" in processed_app_review:
            processed_app_review["_id"] = str(processed_app_review["_id"])
        processed_app_review["full_review"] = (
            app_review["title"] + " " + app_review["body"]).strip()
        processed_app_review["feature_bigram"] = self.vectorizer_bigram.transform(
            [review]).toarray().tolist()[0]
        processed_app_review["feature_bow"] = self.vectorizer_bow.fit_transform(
            [review]).toarray().tolist()[0]
        processed_app_review["feature_keyword_bug"] = review.count("bug")
        processed_app_review["feature_keyword_freeze"] = review.count("freeze")
        processed_app_review["feature_keyword_crash"] = review.count("crash")
        processed_app_review["feature_keyword_glitch"] = review.count("glitch")
        processed_app_review["feature_keyword_wish"] = review.count("wish")
        processed_app_review["feature_keyword_should"] = review.count("should")
        processed_app_review["feature_keyword_add"] = review.count("add")
        processed_app_review["feature_tense_past"] = review_tense.no_past
        processed_app_review["feature_tense_present"] = review_tense.no_present
        processed_app_review["feature_tense_future"] = review_tense.no_future
        processed_app_review["feature_rating"] = app_review["rating"]
        processed_app_review["feature_sentiment_score_pos"] = NLPHelper.get_sentiment_pos_score(
            review_sentiment)
        processed_app_review["feature_sentiment_score_neg"] = NLPHelper.get_sentiment_neg_score(
            review_sentiment)
        processed_app_review["feature_sentiment_score_single"] = NLPHelper.get_sentiment_single_score(
            review_sentiment)
        processed_app_review["feature_word_count"] = NLPHelper.extract_word_cont(
            original_review)
        processed_app_review["feature_contains_keywords_bug"] = NLPHelper.extract_keyword_freq(
            feature="bug", text=original_review)
        processed_app_review["feature_contains_keywords_feature_request"] = NLPHelper.extract_keyword_freq(
            feature="feature", text=original_review)

        return processed_app_review

    @staticmethod
    def get_bow_vectorizer():
        vocabulary_bow = PickleHandler.load_vocabulary(name="bow")

        return CountVectorizer(vocabulary=vocabulary_bow)

    @staticmethod
    def get_bigram_vectorizer():
        vocabulary_bigram = PickleHandler.load_vocabulary(name="bigram")
        vectorizer_bigram = CountVectorizer(ngram_range=(
            2, 2), token_pattern=r'\b\w+\b', min_df=10)
        vectorizer_bigram.fit_transform(vocabulary_bigram).toarray()

        return vectorizer_bigram


############################
# NLP HELPER
############################
CUSTOM_STOPWORDS = ['i', 'me', 'up', 'my', 'myself', 'we', 'our', 'ours',
                    'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
                    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
                    'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
                    'themselves', 'am', 'is', 'are', 'a', 'an', 'the', 'and', 'in', 'of', 'so',
                    'out', 'on', 'up', 'down', 's', 't', 'to', 'be', 'your', 'have', 'app', 'too']
CONTRACTIONS = ["ain't", "aren't", "can't", "can't've", "'cause", "could've", "couldn't", "couldn't've", "didn't",
                "doesn't", "don't", "hadn't", "hadn't've", "hasn't", "haven't", "he'd", "he'd've", "he'll", "he'll've",
                "he's", "how'd", "how'd'y", "how'll", "how's", "i'd", "i'd've", "i'll", "i'll've", "i'm", "i've",
                "isn't", "it'd", "it'd've", "it'll", "it'll've", "it's", "let's", "ma'am", "mayn't", "might've",
                "mightn't", "mightn't've", "must've", "mustn't", "mustn't've", "needn't", "needn't've", "o'clock",
                "oughtn't", "oughtn't've", "shan't", "sha'n't", "shan't've", "she'd", "she'd've", "she'll", "she'll've",
                "she's", "should've", "shouldn't", "shouldn't've", "so've", "so's", "that'd", "that'd've", "that's",
                "there'd", "there'd've", "there's", "they'd", "they'd've", "they'll", "they'll've", "they're",
                "they've", "to've", "wasn't", "we'd", "we'd've", "we'll", "we'll've", "we're", "we've", "weren't",
                "what'll", "what'll've", "what're", "what's", "what've", "when's", "when've", "where'd", "where's",
                "where've", "who'll", "who'll've", "who's", "who've", "why's", "why've", "will've", "won't", "won't've",
                "would've", "wouldn't", "wouldn't've", "y'all", "y'all'd", "y'all'd've", "y'all're", "y'all've",
                "you'd", "you'd've", "you'll", "you'll've", "you're", "you've"]
KEYWORDS_BUG = "bug|crash|glitch|freeze|hang|not work|stop work|kill|dead|frustrate|froze|fix|close|error|gone|problem"
KEYWORDS_FEATURE_REQUEST = "should|wish|add|miss|lack|need"
CONTRACTIONS_EXPANDED = {
    "ain't": "am not",  # are not; is not; has not; have not",
    "aren't": "are not",  # ; am not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had",  # , / he would",
    "he'd've": "he would have",
    "he'll": "he shall",  # / he will",
    "he'll've": "he shall have",  # / he will have",
    "he's": "he has",  # / he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how has",  # / how is / how does",
    "i'd": "i had",  # / i would",
    "i'd've": "i would have",
    "i'll": "i will",  # / i shal",
    "i'll've": "i will have",  # / i shall have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",  # / it had",
    "it'd've": "it would have",
    "it'll": "it will",  # / it shall",
    "it'll've": "it will have",  # / it shall have",
    "it's": "it is",  # / it has",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had",  # / she would",
    "she'd've": "she would have",
    "she'll": "she shall",  # / she will",
    "she'll've": "she shall have",  # / she will have",
    "she's": "she is",  # / she has",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",  # / so as",
    "that'd": "that would",  # / that had",
    "that'd've": "that would have",
    "that's": "that is",  # / that has",
    "there'd": "there had",  # / / there would",
    "there'd've": "there would have",
    "there's": "there is",  # / there has",
    "they'd": "they had",  # / they would",
    "they'd've": "they would have",
    "they'll": "they shall / they will",
    "they'll've": "they will have",  # / they shall have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had ",  # / we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",  # / what will",
    "what'll've": "what will have",  # / what shall have",
    "what're": "what are",
    "what's": "what is",  # / what has",
    "what've": "what have",
    "when's": "when is ",  # / when has",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",  # / where has",
    "where've": "where have",
    "who'll": "who will",  # / who will",
    "who'll've": "who will have ",  # / who will have",
    "who's": "who is",  # / who has",
    "who've": "who have",
    "why's": "why is",  # / why has",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had",  # / you would",
    "you'd've": "you would have",
    "you'll": "you will",  # / you shall",
    "you'll've": "you will have",  # / you shall have",
    "you're": "you are",
    "you've": "you have"
}

l = nltk.WordNetLemmatizer()
t = nltk.RegexpTokenizer('[a-z]\w+')

cwd = os.getcwd()
print('PATH:', cwd)
print('files:', os.listdir('.'))

path_to_model = "stanford-postagger-full-2016-10-31/models/english-bidirectional-distsim.tagger"
path_to_jar = "stanford-postagger-full-2016-10-31/stanford-postagger.jar"
pos_tagger = nltk.StanfordPOSTagger(model_filename=path_to_model, path_to_jar=path_to_jar,
                                    java_options='-Xmx4060m -mx4060m')


class NLPHelper:
    @staticmethod
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return 'a'

    @staticmethod
    def lem(sentence):
        tokens = t.tokenize(sentence)
        result = []
        for (token, tag) in nltk.pos_tag(tokens):
            result.append(l.lemmatize(token, NLPHelper.get_wordnet_pos(tag)))

        return " ".join(result)

    @staticmethod
    def remove_stopwords(text):
        return " ".join([w for w in text.split() if w not in CUSTOM_STOPWORDS])

    @staticmethod
    def remove_tokens(text, tokens):
        return " ".join([w for w in text.split() if w not in tokens])

    @staticmethod
    def expand_contraction(text):
        """expands word if word is a contraction
        :param text to check :returns text with expanded words"""
        expanded_contraction_sentence = ''
        for word in text.split():
            word = word.replace("’", "'")
            if word in CONTRACTIONS_EXPANDED.keys():
                word = CONTRACTIONS_EXPANDED[word]
            expanded_contraction_sentence += (' ' + word)
        return expanded_contraction_sentence.strip()

    @staticmethod
    def remove_duplicated_words(text):
        return ' '.join(OrderedDict((w, w) for w in text.split()).keys())

    @staticmethod
    def pos_tag(text):
        return pos_tagger.tag(text)

    @staticmethod
    def extract_keyword_and_remove_from_text(keywords, text):
        found_keywords = re.findall(keywords, text, re.IGNORECASE)
        cleaned_text = " ".join(
            [w for w in text.split() if w not in found_keywords])
        print("found_keywords", found_keywords,
              " | cleaned text: ", cleaned_text)
        return found_keywords, cleaned_text

    @staticmethod
    def extract_keyword_freq(feature="bug", text=""):
        if feature == "bug":
            return len(re.findall(KEYWORDS_BUG, text, re.IGNORECASE))
        elif feature == "feature":
            return len(re.findall(KEYWORDS_FEATURE_REQUEST, text, re.IGNORECASE))

    @staticmethod
    def keep_only_words_with_pos_tags(review, tags):
        tokens = t.tokenize(review)
        pos_tagged = NLPHelper.pos_tag(tokens)
        output_review = ""
        for (token, pos_tag) in pos_tagged:
            if pos_tag in tags:
                output_review += token + " "
        return output_review

    @staticmethod
    def extract_word_cont(text):
        return len(text.split())

    @staticmethod
    def count_tense_occurrences(review):
        tense = Tense()
        pos_tagged = NLPHelper.pos_tag(review)

        for (token, pos_tag) in pos_tagged:
            if pos_tag == "VB":
                tense.no_present = tense.no_present + 1
            elif pos_tag == "VBD":
                tense.no_past = tense.no_past + 1
            elif pos_tag == "VBG":
                tense.no_present = tense.no_present + 1
            elif pos_tag == "VBN":
                tense.no_past = tense.no_past + 1
            elif pos_tag == "VBP":
                tense.no_present = tense.no_present + 1
            elif pos_tag == "VBZ":
                tense.no_present = tense.no_present + 1

            if token in ["will", "ll", "shall"]:
                tense.no_future = tense.no_future + 1

        return tense

    @staticmethod
    def get_sentiment(review):
        DIR_ROOT = os.getcwd()
        senti_jar = os.path.join(DIR_ROOT, 'sentistrength/SentiStrength.jar')
        senti_folder = os.path.join(
            DIR_ROOT, 'sentistrength/SentStrength_Data/')
        p = subprocess.Popen(
            shlex.split('java -jar ' + senti_jar +
                        ' stdin sentidata ' + senti_folder),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        # communicate via stdin the string to be rated. Note that all spaces are replaced with +
        stdout_text, stderr_text = p.communicate(
            bytearray(review.replace(" ", "+"), 'utf8'))
        p.kill()
        return str(stdout_text, 'utf-8').split()

    @staticmethod
    def get_sentiment_pos_score(sentiment_raw):
        return int(sentiment_raw[0])

    @staticmethod
    def get_sentiment_neg_score(sentiment_raw):
        return int(sentiment_raw[1])

    @staticmethod
    def get_sentiment_single_score(sentiment_raw):
        score_pos = int(sentiment_raw[0])
        score_neg = int(sentiment_raw[1])
        if abs(score_pos) > abs(score_neg):
            return score_pos
        elif abs(score_pos) == abs(score_neg):
            return 0
        else:
            return score_neg


class Tense:
    def __init__(self):
        self.no_present = 0
        self.no_past = 0
        self.no_future = 0


if __name__ == '__main__':
    print(NLPHelper.expand_contraction(
        "cannot able to login.. always the server doesn't work.."))
