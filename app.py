
import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)


st.title("📰 Fake News Detection System")

st.write(
    "Enter a news headline and the trained "
    "machine learning model will classify it."
)


@st.cache_resource
def load_model():

    with open(
        "fake_news_model.pkl",
        "rb"
    ) as file:

        model = pickle.load(file)

    with open(
        "tfidf_vectorizer.pkl",
        "rb"
    ) as file:

        tfidf = pickle.load(file)

    return model, tfidf


model, tfidf = load_model()



stop_words = set(
    stopwords.words('english')
)

lemmatizer = WordNetLemmatizer()


def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(
        r'http\S+|www\S+|https\S+',
        '',
        text
    )

    text = re.sub(
        r'<.*?>',
        '',
        text
    )

    text = re.sub(
        r'[^a-zA-Z\s]',
        '',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    ).strip()

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return ' '.join(words)


news = st.text_area(
    "Enter News Headline",
    placeholder="Example: Government announces new education policy...",
    height=150
)


if st.button(
    "🔍 Check News",
    use_container_width=True
):

    if news.strip() == "":

        st.warning(
            "Please enter a news headline."
        )

    else:

        
        cleaned_news = clean_text(news)


        vector = tfidf.transform(
            [cleaned_news]
        )

        prediction = model.predict(
            vector
        )[0]

        if prediction == 1:

            st.success(
                "✅ Prediction: REAL NEWS"
            )

        else:

            st.error(
                "❌ Prediction: FAKE NEWS"
            )


        with st.expander(
            "View NLP Processed Text"
        ):

            st.write(
                cleaned_news
            )


st.divider()

st.subheader(
    "📌 About This Project"
)

st.write(
    """
    This project uses Natural Language Processing (NLP)
    and Machine Learning to classify news headlines.

    NLP techniques used:
    • Text cleaning
    • Stopword removal
    • Lemmatization

    Feature extraction:
    • TF-IDF

    Machine Learning models:
    • Logistic Regression
    • Naive Bayes
    • Linear SVM
    """
)


# st.info(
#     "Note: This system classifies text according to "
#     "patterns learned from the dataset. It does not "
#     "independently verify news using external sources."
# )