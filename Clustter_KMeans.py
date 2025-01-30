
#----------------------------------------------
# This portion is for Cluster K-Means Algorithm
#----------------------------------------------

df_kmeans["Reviews"] = df_kmeans["Reviews"].values.astype("U")
vectorize = TfidfVectorizer(stop_words='english')
vectorized_value = vectorize.fit_transform(df_kmeans["Reviews"])

k_value = 10
k_model = KMeans(n_clusters=k_value, init='k-means++', max_iter=100, n_init=1)
k_model.fit(vectorized_value)

df_kmeans["clusters"] = k_model.labels_
df_kmeans.head()


# cluster_groupby = df_kmeans.groupby("clusters")
# for cluster in cluster_groupby.groups:
#     f = open("cluster"+str(cluster)+".csv","w")
#     data = cluster_groupby.get_group(cluster)[["Rating", "Reviews"]]
#     f.write(data.to_csv(index_label="id"))
#     f.close()

center_gravity = k_model.cluster_centers_.argsort()[:,::-1]
terms = vectorize.get_feature_names_out()

for ctr in range(k_value):
    print ("Cluster %d: " % ctr)
    for ctr2 in center_gravity[ctr, :10]:
        print ("%s" % terms[ctr2])
    print ("---------------------")

plt.scatter(df_kmeans['Reviews'], df_kmeans['clusters'])
plt.xlabel('clusters')
plt.ylabel('Reviews')
plt.show()