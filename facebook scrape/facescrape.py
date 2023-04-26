import os
from bs4 import BeautifulSoup
from selenium import webdriver
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

def get_soup(url):
    driver = webdriver.Chrome()
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    driver.quit()
    return soup

def extract_posts(soup):
    # You may need to adjust the CSS selector to match the specific page's structure
    posts = []
    for post in soup.find_all('div', {'class': 'post'}):
        content = post.get_text(strip=True)
        posts.append(content)
    return posts

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    filtered_tokens = [stemmer.stem(token) for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(filtered_tokens)

def cluster_topics(texts, n_clusters):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    return kmeans.predict(X)

def main():
    url = 'https://www.facebook.com/patrick.delves.1'
    soup = get_soup(url)
    
    # Step 1: Turn each post into its own topic
    posts = extract_posts(soup)
    
    # Step 2: Save all images & videos for each post that is turned into a topic
    # Skipping this step as media extraction is not applicable in this example
    
    # Step 3: Cluster/group the topics based on the content (text) in each post
    preprocessed_posts = [preprocess_text(post) for post in posts]
    n_clusters = 3 # Adjust this number based on your desired number of clusters
    cluster_labels = cluster_topics(preprocessed_posts, n_clusters)
    
    clustered_posts = defaultdict(list)
    for label, post in zip(cluster_labels, posts):
        clustered_posts[label].append(post)

    for label, posts in clustered_posts.items():
        print(f'Cluster {label}:')
        for post in posts:
            print(f'- {post}')

if __name__ == '__main__':
    main()
