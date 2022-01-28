import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import sys
import pickle
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
#Load movies data
movies = pd.read_csv('raw_data/movies.csv')
genome_scores = pd.read_csv('raw_data/genome-scores.csv')
tags = pd.read_csv('raw_data/tags.csv')
genome_tags = pd.read_csv('raw_data/genome-tags.csv')
#Use ratings data to downsample tags data to only movies with ratings 
ratings = pd.read_csv('raw_data/ratings.csv')
#ratings = ratings.drop_duplicates('movieId')

movies.tail()
movies['genres'] = movies['genres'].str.replace('|',' ')

#limit ratings to user ratings that have rated more that 55 movies -- 
#Otherwise it becomes impossible to pivot the rating dataframe later for collaborative filtering.

# ratings_f = ratings.groupby('userId').filter(lambda x: len(x) >= 55)

# list the movie titles that survive the filtering
# movie_list_rating = ratings_f.movieId.unique().tolist()

#filter the movies data frame
# movies = movies[movies.movieId.isin(movie_list_rating)]

# map movie to id:
Mapping_file = dict(zip(movies.title.tolist(), movies.movieId.tolist()))

tags.drop(['timestamp'],1, inplace=True)
# ratings_f.drop(['timestamp'],1, inplace=True)

mixed = pd.merge(movies, tags, on='movieId', how='left')

# create metadata from tags and genres
mixed.fillna("", inplace=True)
mixed = pd.DataFrame(mixed.groupby('movieId')['tag'].apply(
                                          lambda x: "%s" % ' '.join(x)))
Final = pd.merge(movies, mixed, on='movieId', how='left')
Final ['metadata'] = Final[['tag', 'genres']].apply(
                                          lambda x: ' '.join(x), axis = 1)
Final[['movieId','title','metadata']].head(3)


tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(Final['metadata'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=Final.index.tolist())
print(tfidf_df.shape)

# Compress with SVD
svd = TruncatedSVD(n_components=25)
latent_matrix = svd.fit_transform(tfidf_df)
# plot var expalined to see what latent dimensions to use
explained = svd.explained_variance_ratio_.cumsum()
plt.plot(explained, '.-', ms = 16, color='red')
plt.xlabel('Singular value components', fontsize= 12)
plt.ylabel('Cumulative percent of variance', fontsize=12)        
plt.show()


latent_matrix = preprocessing.scale(latent_matrix)

with open('./processed_data/' + 'Arm_FeatureVectors_2.dat', 'a+') as f:
    f.write('ArticleID')
    f.write('\t'+ 'FeatureVector')
    f.write('\n')

movieIds = Final.movieId.tolist()
    
for i in range(len(movieIds)):
    articleID = movieIds[i]
    featureVector = latent_matrix[i]
    with open('./processed_data/' + 'Arm_FeatureVectors_2.dat', 'a+') as f:
        f.write(str(articleID))
        f.write('\t'+ ';'.join([str(x) for x in featureVector]))
        f.write('\n')