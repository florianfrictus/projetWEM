import pandas as pd
import numpy as np
import re
import contractions
from tools.read import get_data
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

from vaderSentiment_fr.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from textblob_fr import PatternTagger, PatternAnalyzer

french_stopwords = set(stopwords.words('french'))
# %load_ext autoreload
# %autoreload 2


def normalize_lemm_stem(comment):
    """
    Normalize comments
    :param comment: comment to normalize
    :return: normalized comment
    """
    # lowercase
    comment = comment.lower()
    # remove punctuation
    if re.sub(r'[^\w\s]', '', comment) != '': comment = re.sub(r'[^\w\s]', '', comment)
    # replace contractions
    comment = contractions.fix(comment)
    # tokenize sentences
    comment = word_tokenize(comment, language='french')
    # stopwords
    comment = [token for token in comment if token not in french_stopwords]
    # lemmatize
    lemmatizer = WordNetLemmatizer()
    comment = list(map(lemmatizer.lemmatize, comment))
    # stem
    # stemmer = SnowballStemmer("french")
    # comment = list(map(stemmer.stem, comment))
    # length
    comment = [token for token in comment if len(token) > 2]

    return ' '.join(comment)


def get_bombing_words():
    """
    Get dictionary of review bombing words
    :return: list of review bombing words
    """
    with open('Words/review-bombing-words.txt', errors='ignore', encoding="utf-8") as opened:
        contents = opened.read()
    contents_lines = contents.split('\n')

    return [x for x in contents_lines if len(x) > 0]


def convert_to_grade(percentage):
    """
    Convert a range [-1, 1] to a grade [0, 20] where 0 => 12
    :param percentage: range[-1, 1]
    :return: grade[0, 20]
    """
    if percentage < 0:
        grade = np.floor(12 * percentage + 12)
    else:
        grade = np.ceil(8 * percentage + 12)
    return grade


def extreme_behaviour(dataframe, sentiment='positive'):
    """
    Get a list of users with a suspicious behaviour by given always excellent or bad grading
    :param dataframe: dataframe to analyse
    :param sentiment: kind of sentiment to extract 'positive' or 'negative'
    :return: list of users with an extreme behaviour
    """
    df_name = dataframe.groupby('username').mean('grade')
    df_name_count = dataframe.groupby('username').count()
    df_name['count'] = df_name_count['grade']
    try:
        if sentiment == 'positive':
            names = df_name[(df_name['grade'] > 18) & (df_name['count'] > 1)].index.values
        if sentiment == 'negative':
            names = df_name[(df_name['grade'] < 10) & (df_name['count'] > 1)].index.values
        return names
    except:
        print("Extreme Behaviour accepts only 'positive' or 'negative' sentiment")


def naive_bombing(dataframe, sentiment='positive'):
    """
    Filter suspicious comments involved in review bombing process
    :param dataframe: dataframe to analyse
    :param sentiment: kind of sentiment to extract 'positive' or 'negative'
    :return: Dataframe suspected of review bombing
    """
    bombing = []
    bombing_words = get_bombing_words()
    try:
        for row in dataframe.iterrows():
            document_words = set(word for word in word_tokenize(row[1]['comment']))
            bombing_alert = list(document_words.intersection(bombing_words))
            if len(bombing_alert) > 0:
                if sentiment == 'positive' and row[1]['grade'] > 18:
                    bombing.append(row[1])
                if sentiment == 'negative' and row[1]['grade'] < 2:
                    bombing.append(row[1])
        return pd.concat(bombing, axis=1, keys=[s.name for s in bombing]).transpose()
    except:
        print("Naive Bombing accepts only 'positive' or 'negative' sentiment")


def extract_game_sentiment(dataframe, game=None):
    """
    Polarize comments in a range [-1, 1] with Vader and (TextBlob)
    Filter dataframe by a selected game
    :param dataframe: dataframe to polarize
    :param game: (optional) select a specific game
    :return: polarized dataframe
    """
    try:
        if game:
            dataframe = dataframe[dataframe['game'] == game]
        dataframe = sentiment_vader(dataframe=dataframe)
        # dataframe = sentiment_textblob(dataframe=dataframe)
    except:
        print("There is no comment with the game")
    return dataframe


def sentiment_vader(dataframe):
    """
    Polarize comments in a range [-1, 1] with Vader
    :param dataframe: dataframe to polarize
    :return: polarized dataframe
    """
    # Sentiment analysis using Vader range is [-1, 1]
    # https://github.com/cjhutto/vaderSentiment
    senti_vader = [SentimentIntensityAnalyzer().polarity_scores(comment)
                   for comment in dataframe['comment_normalized']]
    dataframe['compound_vader'] = [senti['compound'] for senti in senti_vader]
    return dataframe


def sentiment_textblob(dataframe):
    """
    Polarize comments in a range [-1, 1] with TextBlob
    :param dataframe: dataframe to polarize
    :return: polarized dataframe
    """
    # Sentiment analysis using TextBlob range is [-1, 1]
    # Dedicated for French: https://github.com/sloria/textblob-fr
    sentiment_blob = [TextBlob(comment, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer()).sentiment
                      for comment in dataframe['comment_normalized']]
    dataframe['polarity_tb'] = [senti[0] for senti in sentiment_blob]
    dataframe['subjectivity_tb'] = [senti[1] for senti in sentiment_blob]
    return dataframe


def predict_review_bombing(dataframe, sentiment=None, confidence=None):
    """
    Prediction of review bombing comments for a defined dataframe
    :param dataframe: dataframe for review bombing prediction
    :param sentiment: kind of sentiment to extract 'positive', 'negative' or None
    :param confidence: confidence to detect review bombing 'High', 'Medium', 'Low' or None
    :return: random comment from predicted dataframe
    """
    try:
        if sentiment == 'positive':
            if confidence == 'High':
                comments = dataframe[(dataframe['compound_vader'] < 0.5)]
            else:
                comments = dataframe[(dataframe['compound_vader'] < 0.25)]
        if sentiment == 'negative':
            if confidence == 'High':
                comments = dataframe[(dataframe['compound_vader'] > 0.5)]
            else:
                comments = dataframe[(dataframe['compound_vader'] > 0.25)]
        if not sentiment:
            comments = dataframe
        try:
            random_int = np.random.randint(len(comments))
            return [comments.iloc[random_int]['comment'], comments.iloc[random_int]['username']]
        except:
            return ['There is no {0} comment for this game'.format(sentiment), 'Nobody']
    except:
        print("Predict Review Bombing accepts only 'positive' or 'negative' or 'None' sentiment")


def predict_review_bombing_table(dataframe, sentiment=None, confidence=None):
    """
    Prediction of review bombing comments for a defined dataframe
    :param dataframe: dataframe for review bombing prediction
    :param sentiment: kind of sentiment to extract 'positive' or 'negative'
    :param confidence: confidence to detect review bombing 'High', 'Medium' or 'Low'
    :return: predicted dataframe
    """
    try:
        if sentiment == 'positive':
            if confidence == 'High':
                return dataframe[(dataframe['compound_vader'] < 0.25)]
            else:
                return dataframe[(dataframe['compound_vader'] < 0.5)]
        elif sentiment == 'negative':
            if confidence == 'High':
                return dataframe[(dataframe['compound_vader'] > 0.5)]
            else:
                return dataframe[(dataframe['compound_vader'] > 0.25)]
        else:
            return dataframe
    except:
        print("Predict Review Bombing Table accepts only 'positive' or 'negative' or 'None' sentiment")


if __name__ == "__main__":
    # Load dataset
    dataset = get_data('data/dataset500.csv')
    data = [{'game': dataset['name'][i], 'platform': dataset['platform'][i],
             'grade': comment['grade'][0], 'comment': comment['comment'][0], 'username': comment['username'][0]}
            for i, comments in enumerate(dataset['comments']) for comment in comments]
    df = pd.DataFrame(data, columns=['game', 'platform', 'grade', 'comment', 'username'])
    # Normalize comments
    df['comment_normalized'] = [normalize_lemm_stem(comment) for comment in df['comment']]
    # Process the naive bombing
    review_pos = naive_bombing(dataframe=df, sentiment='positive')
    # Extract sentiment for a define game
    review_pos = extract_game_sentiment(dataframe=review_pos, game='Gran Turismo 7')
    # Get users with extreme behaviour
    names = extreme_behaviour(dataframe=df, sentiment='positive')
    review_pos = review_pos[review_pos['username'].isin(names)]
    print(review_pos)
    # Predict review bombing comment
    pred = predict_review_bombing(dataframe=review_pos, sentiment='positive')
    print(pred[0])
