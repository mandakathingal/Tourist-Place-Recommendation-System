from django.db import models
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import joblib
from django.shortcuts import render


def get_recommendations(selected_place_name):
    data = pd.read_csv(r"C:\Users\ADMIN\Desktop\project\recomendation\csv file1.csv", error_bad_lines=False, encoding="latin-1")
    data = data.drop(["Review", "Name", "Date", "Raw_Review", "City.1", "Unnamed: 9"], axis=1)

    combine_place_rating = data.dropna(axis=0, subset=['Place'])
    place_ratingCount = combine_place_rating.groupby(by=['Place'])['Rating'].count().reset_index().rename(columns={'Rating': 'totalRatingCount'})[['Place', 'totalRatingCount']]

    rating_with_totalRatingCount = combine_place_rating.merge(place_ratingCount, left_on='Place', right_on='Place', how='left')

    popularity_threshold = 10
    rating_popular_place = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')

    place_features_df = rating_popular_place.pivot_table(index='Place', columns='userId', values='Rating').fillna(0)
    place_features_df_matrix = csr_matrix(place_features_df.values)

    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(place_features_df_matrix)

    try:
        query_index = place_features_df.index.get_loc(selected_place_name)
    except KeyError:
        print("Invalid place name. Please try again.")
        exit()

    distances, indices = model_knn.kneighbors(place_features_df.iloc[query_index, :].values.reshape(1, -1), n_neighbors=6)

    recommendations = []
    for i in range(0, len(distances.flatten())):
        if i == 0:
            recommendations.append('Recommendations for {0}:'.format(selected_place_name))
        else:
            recommendations.append('{0}: {1}'.format(i, place_features_df.index[indices.flatten()[i]]))

    return recommendations
