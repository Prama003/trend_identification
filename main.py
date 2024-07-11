import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud

# Step 1: Load the Dataset
# Replace with the path to your dataset
data = {
    'date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'text': [
        'I love the new features of this product!', 'This brand is the worst ever.', 
        'Had a great experience with the customer service.', 'Not satisfied with the quality.',
        'The product is amazing!', 'I hate the new update.', 'This is okay, but could be better.',
        'Excellent product, highly recommend!', 'Terrible experience, will not buy again.',
        'The new design is fantastic!'
    ] * 10,
    'topic': ['Product A'] * 50 + ['Product B'] * 50
}

df = pd.DataFrame(/content/twitter_training)

# Step 2: Data Cleaning and Preprocessing
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = re.sub(r'\d+', '', text)  # Remove numbers
    return text

df['clean_text'] = df['text'].apply(clean_text)

# Step 3: Perform Sentiment Analysis
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    scores = sia.polarity_scores(text)
    return scores['compound']

df['sentiment_score'] = df['clean_text'].apply(analyze_sentiment)
df['sentiment'] = df['sentiment_score'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))

# Step 4: Visualize Sentiment Patterns
# Bar chart of sentiment distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='sentiment', data=df)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Time series plot of sentiment over time
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
sentiment_over_time = df.resample('W').mean()

plt.figure(figsize=(10, 6))
plt.plot(sentiment_over_time.index, sentiment_over_time['sentiment_score'])
plt.title('Sentiment Over Time')
plt.xlabel('Date')
plt.ylabel('Average Sentiment Score')
plt.show()

# Word cloud of positive and negative words
positive_text = ' '.join(df[df['sentiment'] == 'positive']['clean_text'])
negative_text = ' '.join(df[df['sentiment'] == 'negative']['clean_text'])

positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
negative_wordcloud = WordCloud(width=800, height=400, background_color='black').generate(negative_text)

plt.figure(figsize=(15, 8))
plt.subplot(1, 2, 1)
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Positive Words')

plt.subplot(1, 2, 2)
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Negative Words')

plt.show()

# Sentiment analysis by topic/brand
plt.figure(figsize=(10, 6))
sns.boxplot(x='topic', y='sentiment_score', data=df)
plt.title('Sentiment by Topic/Brand')
plt.xlabel('Topic/Brand')
plt.ylabel('Sentiment Score')
plt.show()
