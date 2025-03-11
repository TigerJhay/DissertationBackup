import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

#----------------------------------------------
# This portion is for Cluster K-Means Algorithm
#----------------------------------------------

#df = pd.read_csv("D:\TestData.csv")
df = pd.read_csv("D:\My Documents\~Dissertation Files\TestData.csv")
df = df.iloc[:10000,:]
df['Score'].value_counts().plot(kind='bar')
df_reco = df[['Id', 'ProductId', 'Score']]
pivot_table = df_reco.pivot_table(index='Id', columns='ProductId', values='Score', fill_value=0)

num_clusters = 5  # Choose the number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(pivot_table)
user_id = 1

user_cluster_label = cluster_labels[user_id - 1]
users_in_same_cluster = pivot_table.index[cluster_labels == user_cluster_label]
average_ratings = pivot_table.loc[users_in_same_cluster].mean()
sorted_ratings = average_ratings.sort_values(ascending=False)

# Example: Get top-k recommendations
k = 3
top_k_recommendations = sorted_ratings.head(k)

# Print the top-k recommendations
print("Top", k, "recommendations")
for product_id, rating in top_k_recommendations.items():
    print("Product ID:", product_id, "Rating:", rating)
