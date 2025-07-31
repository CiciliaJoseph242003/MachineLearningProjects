

from google.colab import files
uploaded=files.upload()

import pandas as pd
import io

df=pd.read_csv(io.BytesIO(uploaded['Dataset  (1).csv']))

drop_cols = ['Restaurant ID', 'Address', 'Locality',
             'Locality Verbose', 'Switch to order menu', 'Rating color', 'Rating text']

df = df.drop(columns=drop_cols, errors='ignore')

df = df.dropna(subset=['Restaurant Name','Cuisines', 'Average Cost for two', 'Price range', 'Aggregate rating'])

def combine_features(row):
    return f"{row['Cuisines']} {row['Average Cost for two']} {row['Price range']} {row['Aggregate rating']}"

df['combined_features']=df.apply(combine_features,axis=1)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer()
feature_matrix=vectorizer.fit_transform(df['combined_features'])

from sklearn.metrics.pairwise import cosine_similarity

def recommend_restaurants_optimized(cuisine, top_n=5):
    cuisine = cuisine.lower()
    df['match'] = df['Cuisines'].str.lower().str.contains(cuisine)

    matched_df = df[df['match']]
    if matched_df.empty:
        return f"No restaurants found with cuisine containing '{cuisine}'."

    matched_df = matched_df.head(100)
    matched_indices = matched_df.index.tolist()

    sub_matrix = feature_matrix[matched_indices]
    cosine_sim_subset = cosine_similarity(sub_matrix)

    idx = 0
    similarity_scores = list(enumerate(cosine_sim_subset[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    top_indices = [matched_indices[i[0]] for i in similarity_scores[1:top_n+1]]

    # Check if Restaurant Name column exists
    columns_to_show = ['Cuisines', 'Average Cost for two', 'Price range', 'Aggregate rating']
    if 'Restaurant Name' in df.columns:
        columns_to_show = ['Restaurant Name'] + columns_to_show

    recommendations = df.loc[top_indices][columns_to_show]
    return recommendations.reset_index(drop=True)

cuisine_choice = input("Enter a cuisine (e.g., Italian, Chinese): ").strip()
top_n_input = input("How many recommendations do you want? (default = 5): ").strip()
top_n = int(top_n_input) if top_n_input.isdigit() else 5

recommendations = recommend_restaurants_optimized(cuisine_choice, top_n)
print(f"\nTop {top_n} restaurants serving '{cuisine_choice.title()}' cuisine:")
display(recommendations)
