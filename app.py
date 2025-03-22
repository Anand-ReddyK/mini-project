from flask import Flask, request, jsonify
import pandas as pd
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
import joblib
from collections import Counter

# Load the trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

app = Flask(__name__)

# Function to fetch YouTube comments
def get_youtube_comments(video_url, max_comments=20):
    downloader = YoutubeCommentDownloader()
    comments = downloader.get_comments_from_url(video_url, sort_by=SORT_BY_POPULAR)
    return [c['text'] for c in comments][:max_comments]

# Function to predict sentiment
def predict_sentiment(comment):
    comment_tfidf = vectorizer.transform([comment])
    prediction = model.predict(comment_tfidf)
    return prediction[0]

@app.route('/predict', methods=['GET'])
def predict():
    video_url = request.args.get('url')
    
    if not video_url:
        return jsonify({"error": "Missing YouTube URL"}), 400

    # Fetch comments
    comments = get_youtube_comments(video_url)
    if not comments:
        return jsonify({"error": "No comments found"}), 404

    # Predict sentiment for each comment
    predictions = [predict_sentiment(c) for c in comments]

    # Count the number of each sentiment
    sentiment_counts = dict(Counter(predictions))

    # Prepare response
    response = {
        "total_comments": len(comments),
        "statistics": {
            "positive": sentiment_counts.get("positive", 0),
            "neutral": sentiment_counts.get("neutral", 0),
            "negative": sentiment_counts.get("negative", 0),
        },
        "predictions": [{"comment": c, "sentiment": s} for c, s in zip(comments, predictions)]
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
