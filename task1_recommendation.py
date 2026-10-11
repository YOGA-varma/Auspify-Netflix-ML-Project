
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load the dataset
df = pd.read_csv("DATASET.csv")

print("Dataset loaded successfully!")
print("Total titles:", len(df))

# Step 2: Prepare the text features
features = (
    df["listed_in"].fillna("") + " " +
    df["director"].fillna("") + " " +
    df["country"].fillna("") + " " +
    df["rating"].fillna("")
)

# Step 3: Convert text into numerical features
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(features)

print("TF-IDF feature matrix shape:", tfidf_matrix.shape)

# Step 4: Create the recommendation function
def recommend_titles(title, number_of_recommendations=5):

    # Find the selected title
    matches = df.index[
        df["title"].str.lower() == title.lower()
    ].tolist()

    if not matches:
        print(f"\nTitle '{title}' was not found.")
        print("Please enter a title exactly as it appears in the dataset.")
        return None

    title_index = matches[0]

    # Calculate similarity with all other titles
    similarity_scores = cosine_similarity(
        tfidf_matrix[title_index],
        tfidf_matrix
    ).flatten()

    # Sort titles from most to least similar
    similar_indices = similarity_scores.argsort()[::-1]

    # Exclude the selected title itself
    similar_indices = [
        i for i in similar_indices if i != title_index
    ]

    # Select the top recommendations
    top_indices = similar_indices[:number_of_recommendations]

    recommendations = df.iloc[top_indices][
        ["title", "type", "release_year", "listed_in"]
    ].copy()

    recommendations["similarity_score"] = [
        round(float(similarity_scores[i]), 3)
        for i in top_indices
    ]

    return recommendations

# Step 5: Test the recommendation system
selected_title = df.iloc[0]["title"]

print("\nSelected Netflix title:", selected_title)
print("\nRecommended titles:")

results = recommend_titles(selected_title, 5)

if results is not None:
    print(results.to_string(index=False))

    # Step 6: Save the results
    results.to_csv(
        "outputs/task1_recommendations.csv",
        index=False
    )

    print("\nRecommendations saved to outputs/task1_recommendations.csv")
