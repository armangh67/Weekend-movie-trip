# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 11:39:43 2019

@author: a454g185
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
movies_df = pd.read_csv('https://raw.githubusercontent.com/armangh67/Weekend-movie-trip/master/datasets/movies.csv', low_memory = False)
movies_df.head(10)
tags_df = pd.read_csv('https://raw.githubusercontent.com/armangh67/Weekend-movie-trip/master/datasets/tags.csv' , low_memory = False)
tags_df.head(10)
rating_df = pd.read_csv('https://raw.githubusercontent.com/armangh67/Weekend-movie-trip/master/datasets/ratings.csv' , low_memory = False)
rating_df.head(10)
merged_df=pd.merge(rating_df,tags_df, on = ['userId','movieId'] , how ='left')
merged_df=pd.merge(merged_df , movies_df , on = ['movieId'] , how ='left')
merged_df.head(10)

#Number of user ratings per movie 
userRatings=merged_df[['movieId','userId']].groupby('movieId',as_index=False).count().rename(columns={'userId':'numberOfUserRatings'})
userRatings.head()
tag_counts = merged_df['tag'].value_counts()
#tag_counts[:10].plot(kind='bar', figsize=(10,5))

#print(pd.isnull(merged_df).any())
merged=merged_df.loc[:,['userId','movieId' , 'title', 'rating' , 'genres']]
#merged.boxplot(column='rating', figsize=(10,5), return_type='axes')
#To Count the number of movies in each genres
def count_genres(Data, column, liste):
    genres_count = {}
    for s in liste: 
        genres_count[s] = 0
    for words in Data[column].str.split('|'):
        if type(words) == float and pd.isnull(words):
            continue
        for s in words: 
            if pd.notnull(s):
                genres_count[s] += 1
    # convert the dictionary in a list to sort the keywords  by frequency
    genres_occurences = []
    for k,v in genres_count.items():
        genres_occurences.append([k,v])
    genres_occurences.sort(key = lambda x:x[1], reverse = True)
    return genres_occurences, genres_count
#here we  make census of the genres:
genre_labels = set()
for s in merged_df['genres'].str.split('|').values:
    genre_labels = genre_labels.union(set(s))
    
#counting how many times each of genres occur:
genres_occurences, dum = count_genres(merged_df, 'genres', genre_labels)
# Graph the Genres vs Occurrences
fig = plt.figure(1, figsize=(15,15))
ax2 = fig.add_subplot(2,1,2)
y_axis = [i[1] for i in genres_occurences]
x_axis = [k for k,i in enumerate(genres_occurences)]
x_label = [i[0] for i in genres_occurences]
plt.xticks(rotation=85, fontsize = 15)
plt.yticks(fontsize = 15)
plt.xticks(x_axis, x_label)
plt.ylabel("No. of Genres occurences", fontsize = 24, labelpad = 0)
ax2.bar(x_axis, y_axis, align = 'center',color='r')

plt.title("Genres",bbox={'facecolor':'k', 'pad':5},color='w',fontsize = 25)
plt.show()

new=merged_df.loc[:,['movieId' , 'title', 'genres', 'userId','rating' ]]

# Calculating the average ratings for each movie 
avgratings_df = new.groupby('movieId')['rating'].mean().reset_index(name ='Average Rating')

df1 = merged_df.loc[:,['movieId' , 'title', 'genres']]
df1 = df1.drop_duplicates('movieId')
df1 = pd.merge(avgratings_df,df1, on = ['movieId'])
final_df = df1.loc[:,['movieId','Average Rating','genres']]

from sklearn.preprocessing import LabelEncoder
object1 = LabelEncoder()

final_df['genres'] = object1.fit_transform(final_df['genres'].astype('str'))



X=final_df.sample(1000)
X = X.iloc[:,[1,2]].values
from sklearn.cluster import KMeans
wcss =[]
for i in range (1,10):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter =300, n_init = 10, random_state = 0)
    kmeans.fit(final_df)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,10),wcss)
plt.title('Elbow Method')
plt.xlabel('No of clusters')
plt.ylabel('WCSS')
plt.show()


kmeans=KMeans(n_clusters= 4, init = 'k-means++', max_iter = 1000, n_init = 10, random_state = 0)
Y_Kmeans = kmeans.fit_predict(X)


plt.figure(figsize=(10,10))
plt.scatter(X[Y_Kmeans == 0, 0], X[Y_Kmeans == 0,1],s = 10, c='blue', label = 'C1')

plt.scatter(X[Y_Kmeans == 1, 0], X[Y_Kmeans == 1,1],s = 10, c='red', label = 'C2')

plt.scatter(X[Y_Kmeans == 2, 0], X[Y_Kmeans == 2,1],s = 10, c='cyan', label = 'C3')

plt.scatter(X[Y_Kmeans == 3, 0], X[Y_Kmeans == 3,1],s = 10, c='green', label = 'C4')

#plt.scatter(X[Y_Kmeans == 4, 0], X[Y_Kmeans == 4,1],s = 10, c='magenta', label = 'C5')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 10, c = 'black', label = 'Centroids')
   
plt.title('Clusters of Movies')
plt.xlabel('Average Rating')
plt.ylabel('Genre')
plt.legend()
plt.show()