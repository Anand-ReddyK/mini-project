from flask import Flask, request, render_template
import pandas as pd
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
import joblib
from collections import Counter

# Load the trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

app = Flask(__name__)

# Function to fetch YouTube comments
def get_youtube_comments(video_url, max_comments=10):
    downloader = YoutubeCommentDownloader()
    comments = downloader.get_comments_from_url(video_url, sort_by=SORT_BY_POPULAR)
    return [c['text'] for c in comments][:max_comments]

# Function to predict sentiment
def predict_sentiment(comment):
    comment_tfidf = vectorizer.transform([comment])
    prediction = model.predict(comment_tfidf)
    return prediction[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video_url = request.form.get('url')
        
        if not video_url:
            return render_template('result.html', error="Missing YouTube URL")

        # Fetch comments
        comments = get_youtube_comments(video_url)
        if not comments:
            return render_template('result.html', error="No comments found")

        # Predict sentiment for each comment
        predictions = [predict_sentiment(c) for c in comments]

        # Count the number of each sentiment
        sentiment_counts = dict(Counter(predictions))

        # Prepare response data
        return render_template(
            'result.html',
            total_comments=len(comments),
            positive=sentiment_counts.get("positive", 0),
            neutral=sentiment_counts.get("neutral", 0),
            negative=sentiment_counts.get("negative", 0),
            comments=zip(comments, predictions),
            video_url=video_url
        )

    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
