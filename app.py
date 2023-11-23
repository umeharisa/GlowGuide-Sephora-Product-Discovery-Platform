from flask import Flask, request, render_template, jsonify
import pickle
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load necessary objects from pickle files
# (Make sure your files tfidf_objects.pkl and combined_data.pkl are in the correct path)
with open('tfidf_objects.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)
    tfidf_matrix_positive = pickle.load(file)

with open('combined_data.pkl', 'rb') as file:
    combined_data = pickle.load(file)
    df_selectedCols = combined_data['dataframe']
    cluster_names = combined_data['cluster_names']


# Assuming 'positive_products' DataFrame contains positive sentiment products sorted by average rating
# Replace this assumptionSS with loading your actual data

@app.route('/')
def index():
    return render_template('index.html')  # Renders index.html located in templates folder

@app.route('/recommend', methods=['POST'])
def recommend_products():
    # Get user input
    user_concerns = request.form.get('concerns')  # Assuming the input field in index.html has 'concerns' name

    # Transform user concerns to TF-IDF features
    user_concerns_vector = tfidf_vectorizer.transform([user_concerns])

    # Calculate cosine similarity between user's concerns and product concerns for positive sentiment products
    cosine_similarities = linear_kernel(user_concerns_vector, tfidf_matrix_positive)

    # Get the product indices sorted by similarity
    product_indices = cosine_similarities.argsort()[0][::-1]

    # Initialize variables to keep track of recommended products and their details
    recommended_products = []
    recommended_product_names = set()  # Using a set to track product names for uniqueness
    count = 0
    top_n = 5  # Number of recommendations to provide

    # Loop through product indices and recommend top N products with the highest ratings
    # Replace 'positive_products' with your actual product DataFrame
    for idx in product_indices:
        product_name = df_selectedCols['product_name'].iloc[idx]
        price = df_selectedCols['price_usd'].iloc[idx]
        average_rating = df_selectedCols['avg_rating'].iloc[idx]
        image = df_selectedCols['image'].iloc[idx]

        # Check if the product name is not already recommended to avoid duplicates
        if product_name not in recommended_product_names:
            recommended_products.append({
                'product_name': product_name,
                'price': price,
                'average_rating': average_rating,
                'image': image
            })

            recommended_product_names.add(product_name)  # Add product name to the set
            count += 1

        if count == top_n:
            break

    # Pass recommended products to the recommendation.html template for display
    return render_template('recommendations.html', recommended_products=recommended_products)

if __name__ == '__main__':
    app.run(debug=True)
