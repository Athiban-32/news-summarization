import streamlit as st
from PIL import Image
from bs4 import BeautifulSoup as soup
from urllib.request import urlopen
from newspaper import Article
import io
import nltk
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from streamlit_star_rating import st_star_rating
from pymongo import MongoClient
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
st.set_page_config(page_title='News Summarization', page_icon='./Meta/newspaper.ico')

#client = MongoClient('mongodb://localhost:27017/')
client = MongoClient('mongodb+srv://athibanp2015:WF9VI1pIBz6CzBhv@cluster0.2ymgp0p.mongodb.net/')

db = client['news']

collection_save = db['save']
collection_feedback = db['feedback']

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    if sentiment_score > 0:
        return "Positive"
    elif sentiment_score < 0:
        return "Negative"
    else:
        return "Neutral"

def extract_keywords(text):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized_words = [wordnet_lemmatizer.lemmatize(word) for word in filtered_words]
    word_freq = Counter(lemmatized_words)
    top_keywords = word_freq.most_common(5)
    return [keyword for keyword, _ in top_keywords]

def fetch_news_search_topic(topic):
    site = 'https://news.google.com/rss/search?q={}'.format(topic)
    op = urlopen(site)
    rd = op.read()
    op.close()
    sp_page = soup(rd, 'xml')
    news_list = sp_page.find_all('item')
    return news_list

def fetch_top_news():
    site = 'https://news.google.com/news/rss'
    op = urlopen(site)
    rd = op.read()
    op.close()
    sp_page = soup(rd, 'xml')
    news_list = sp_page.find_all('item')
    return news_list

def fetch_category_news(topic):
    site = 'https://news.google.com/news/rss/headlines/section/topic/{}'.format(topic)
    op = urlopen(site)
    rd = op.read()
    op.close()
    sp_page = soup(rd, 'xml')
    news_list = sp_page.find_all('item')
    return news_list

def fetch_news_poster(poster_link):
    try:
        u = urlopen(poster_link)
        raw_data = u.read()
        if raw_data:
            image = Image.open(io.BytesIO(raw_data))
            st.image(image, use_column_width=True)
        else:
            st.warning("No image data found at the provided link.")
    except Exception as e:
        st.error(f"Error fetching image: {e}")
        st.image(Image.open('./Meta/no_image.jpg'), use_column_width=True)

def display_feedback_form(news_title, key_suffix):
    st.subheader("Feedback for: " + news_title)
    rating = st_star_rating("Rate this article", maxValue=5, defaultValue=1,key="rating-" + key_suffix)
    feedback_comment = st.text_area("Provide your comments or suggestions", key="comment-" + key_suffix, height=100)
    submit_feedback = st.button("Submit Feedback", key="submit-" + key_suffix)

    if submit_feedback:
        # Store the feedback data in MongoDB
        rating_value = st.session_state["rating-" + key_suffix]
        comment_value = st.session_state["comment-" + key_suffix]
        store_feedback(news_title, rating_value, comment_value)

def store_feedback(title, rating, comment):
    feedback_data = {
        "title": title,
        "rating": rating,
        "comment": comment
    }
    collection_feedback.insert_one(feedback_data)
    st.success("Feedback submitted successfully!")

def display_news(list_of_news, news_quantity):
    c = 0
    text_for_wordcloud = ""
    for news in list_of_news:
        c += 1
        st.write('**({}) {}**'.format(c, news.title.text))
        news_data = Article(news.link.text)
        try:
            news_data.download()
            news_data.parse()
            news_data.nlp()
            news_text = news_data.summary

            # Apply LSA
            vectorizer = TfidfVectorizer(stop_words='english')
            X = vectorizer.fit_transform([news_text])
            svd = TruncatedSVD(n_components=5)
            svd.fit(X)
            terms = vectorizer.get_feature_names_out()
            st.write("LSA Topics:")
            for i, comp in enumerate(svd.components_):
                terms_comp = zip(terms, comp)
                sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:6]
                st.write(f"Topic {i + 1}: ", [term[0] for term in sorted_terms])

        except Exception as e:
            st.error(e)
        
        with st.expander(news.title.text):
            st.markdown('''<h6 style='text-align: justify;'>{}"</h6>'''.format(news_data.summary),
                        unsafe_allow_html=True)
            st.markdown("[Read more at {}...]({})".format(news.source.text, news.link.text))
        
            sentiment = analyze_sentiment(news_data.summary)
            st.write("Sentiment: ", sentiment)
        
            keywords = extract_keywords(news_data.summary)
            st.write("Keywords: ", ", ".join(keywords))
        
            display_feedback_form(news.title.text, str(c))
        
            if st.button(f"Save {chr(128190)}"+str(c)): 
                save_article(news.title.text, news_data.summary, news.source.text, news.link.text)
        
            text_for_wordcloud += news_data.summary
            
        st.success("Published Date: " + news.pubDate.text)
        if c >= news_quantity:
            break

    st.subheader("Word Cloud")
    generate_word_cloud(text_for_wordcloud)

def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)


def save_article(title, summary, source, link):
    article_data = {
        "title": title,
        "summary": summary,
        "source": source,
        "link": link
    }
    collection_save.insert_one(article_data)
    st.success("Article saved successfully!")

def run():
    st.title('_Feedback_ _Based_ _News_ _Summarization_ 📰🗞️')
    st.subheader('_News Summarization_')

    image = Image.open('./Meta/newspaper2.png')

    col1, col2, col3 = st.columns([3, 5, 3])

    with col1:
        st.write("")

    with col2:
        st.image(image, use_column_width=False)

    with col3:
        st.write("")

    category = ['--Select--', 'Trending News', 'Favourite Topics', 'Search Topic']
    cat_op = st.selectbox('Select your Category', category)
    if cat_op == category[0]:
        st.warning('Please select Type!!')
    elif cat_op == category[1]:
        st.subheader("Here is the Trending news")
        no_of_news = st.slider('Number of News:', min_value=5, max_value=100, step=20)
        news_list = fetch_top_news()
        display_news(news_list, no_of_news)
    elif cat_op == category[2]:
        av_topics = ['Choose Topic', 'WORLD', 'NATION', 'BUSINESS', 'TECHNOLOGY', 'ENTERTAINMENT', 'SPORTS', 'SCIENCE',
                     'HEALTH']
        st.subheader("Choose your favourite Topic")
        chosen_topic = st.selectbox("Choose your favourite Topic", av_topics)
        if chosen_topic == av_topics[0]:
            st.warning("Please Choose the Topic")
        else:
            no_of_news = st.slider('Number of News:', min_value=5, max_value=100, step=20)
            news_list = fetch_category_news(chosen_topic)
            if news_list:
                st.subheader("Here are the some {} News".format(chosen_topic))
                display_news(news_list, no_of_news)
            else:
                st.error("No News found for {}".format(chosen_topic))

    elif cat_op == category[3]:
        user_topic = st.text_input("Enter your Topic")
        no_of_news = st.slider('Number of News:', min_value=5, max_value=100, step=20)

        if st.button("Search") and user_topic != '':
            user_topic_pr = user_topic.replace(' ', '')
            news_list = fetch_news_search_topic(topic=user_topic_pr)
            if news_list:
                st.subheader("Here are the some {} News".format(user_topic.capitalize()))
                display_news(news_list, no_of_news)
            else:
                st.error("No News found for {}".format(user_topic))
        else:
            st.warning("Please write Topic Name to Search")
    
    display_user_feedback()

def display_user_feedback():
    st.title("User Feedback Analysis")
    
    feedback_data = collection_feedback.find({})
    
    st.subheader("Feedback Data")
    if feedback_data:
        feedback_list = [feedback for feedback in feedback_data]
        df_feedback = pd.DataFrame(feedback_list)
        df_feedback = df_feedback.drop(columns=['_id'])
        st.write(df_feedback)
    else:
        st.write("No feedback data available.")

run()
