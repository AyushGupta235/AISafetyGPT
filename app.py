import os
from datetime import datetime, timedelta
from typing import Dict

import pandas as pd
import plotly.express as px
import streamlit as st
from tweety.bot import Twitter

from sentiment_analyzer import analyze_sentiment, create_df_from_tweets

twitter = Twitter()

def on_add_author():
    twitter_handle = st.session_state.twitter_handle
    if twitter_handle.startswith("@"):
        twitter_handle = twitter_handle[1:]
    if twitter_handle in st.session_state.twitter_handles:
        return
    
    all_tweets = twitter.get_tweets(username=twitter_handle)
    if len(all_tweets) == 0:
        return
    st.session_state.twitter_handles[twitter_handle] = all_tweets[0].author.name
    st.session_state.tweets.extend(all_tweets)
    st.session_state.author_sentiment[twitter_handle] = analyze_sentiment(
        twitter_handle=twitter_handle, tweets=st.session_state.tweets
    )

def create_sentiment_df(sentiment_data: Dict[str, int]) -> pd.DataFrame:
    date_list = pd.date_range(
        datetime.now().date() - timedelta(days=6), periods=7, freq="D"
    )
    dates = [str(date) for date in date_list.date]
    chart_data = {"date": dates}

    for author, sentiment in sentiment_data.items():
        author_sentiment = []
        
        for date in dates:
            if date in sentiment:
                author_sentiment.append(sentiment[date])
            else:
                author_sentiment.append(None)
        
        chart_data[author] = author_sentiment
    
    sentiment_df = pd.DataFrame(chart_data)
    sentiment_df.set_index("date", inplace=True)

    if not sentiment_df.empty:
        sentiment_df["Overall"] = sentiment_df.mean(axis=1)
    
    return sentiment_df


st.set_page_config(
    layout="wide",
    page_title="AI Safety Sentiment Analysis",
    page_icon="🤖"
)

st.markdown(
    "<h1 style='text-align: center; color: #ffffff;'>AI Safety GPT</h1>",
    unsafe_allow_html=True
)

if not "tweets" in st.session_state:
    st.session_state.tweets = []
    st.session_state.twitter_handles = {}
    st.session_state.api_key = ""
    st.session_state.author_sentiment = {}

os.environ["OPENAI_API_KEY"] = st.session_state.api_key

col1, col2 = st.columns(2)

with col1:
    st.text_input(
        "OpenAI API Key",
        type="password",
        key="api_key",
        placeholder="Enter your OpenAI API key here",
        help="You can find your API key at https://platform.openai.com/account/api-keys"
    )

    with st.form(key = "twitter_handle_form", clear_on_submit=True):
        st.subheader("Add Twitter Handle", anchor=False)
        st.text_input(
            "Twitter Handle", value="", key="twitter_handle", placeholder="@openai"
        )
        submit = st.form_submit_button(label="Add Tweets", on_click=on_add_author)
        
    if st.session_state.twitter_handles:    
        st.subheader("Twitter Handles", anchor=False)
        for handle, name in st.session_state.twitter_handles.items():
            handle = "@" + handle
            st.markdown(f"{name} ([{handle}](https://twitter.com/{handle}))")

    st.subheader("Tweets", anchor=False)
    st.dataframe(
        create_df_from_tweets(st.session_state.tweets), use_container_width=True
    )

with col2:
    sentiment_df = create_sentiment_df(st.session_state.author_sentiment)
    if not sentiment_df.empty:
        fig = px.line(
            sentiment_df,
            x = sentiment_df.index,
            y = sentiment_df.columns,
            labels = {"date": "Date", "value": "Sentiment"},
        )
        fig.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(fig, theme = "streamlit", use_container_width = True)

        st.dataframe(sentiment_df, use_container_width=True)
