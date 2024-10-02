from flask import Flask, request, render_template
from transformers import pipeline
import pandas as pd

app = Flask(__name__)

# Load the sentiment analysis pipeline
sentiment_analyzer = pipeline('sentiment-analysis')

def process_file(file):
    # Read the file based on its extension
    if file.filename.endswith('.csv'):
        data = pd.read_csv(file)
    elif file.filename.endswith('.xlsx'):
        data = pd.read_excel(file)
    else:
        return None, 'Invalid file format. Only CSV and XLSX are supported.'

    # Check if the 'Review' column exists
    if 'Review' not in data.columns:
        return None, 'The file must contain a "Review" column.'
    
    return data['Review'].tolist(), None

def analyze_sentiments(reviews):
    sentiments = {'positive': 0, 'negative': 0, 'neutral': 0}

    for review in reviews:
        try:
            # Analyze sentiment
            result = sentiment_analyzer(review)
            sentiment = result[0]['label'].lower()

            # Update the sentiment counts
            if sentiment == 'positive':
                sentiments['positive'] += 1
            elif sentiment == 'negative':
                sentiments['negative'] += 1
            else:
                sentiments['neutral'] += 1
        except Exception as e:
            print(f'Error processing review: {e}')
            continue
            
    return sentiments

@app.route('/', methods=['GET', 'POST'])
def sentiment_analysis():
    sentiment_results = None
    error = None

    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            error = 'No file uploaded.'
        else:
            file = request.files['file']

            # Process the file to extract reviews
            reviews, error = process_file(file)
            if not error:
                # Analyze sentiments of the reviews
                sentiment_results = analyze_sentiments(reviews)

    # Render the template, passing sentiment_results and error
    return render_template('index.html', sentiment_results=sentiment_results, error=error)

if __name__ == '__main__':
    app.run(debug=True)
