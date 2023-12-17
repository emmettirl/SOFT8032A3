#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on a cloudy day


Name: Emmett Fitzharris
Student ID: R00222357
Cohort: evSD3

"""

import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Use the below few lines to ignore the warning messages

warnings.filterwarnings("ignore", category=DeprecationWarning)


def warn(*args, **kwargs):
    pass


def main():
    warnings.warn = warn

    df = pd.read_csv("weather.csv")
    # print(df.head())

    # task1(df)
    # task2(df)
    # task3(df)
    # task4(df)
    # task5(df)
    task6(df)
    # task7(df)


def task1(df):
    t1_df = df.copy(deep=True)

    # column 2
    col2 = df.iloc[:, 1]

    # data cleaning
    col2_clean = col2.str.lower()

    # print number of unique locations
    print(col2_clean.nunique())

    col2_vc = col2_clean.value_counts()
    col2_vc_sorted = col2_vc.sort_values(ascending=False)
    vc_tail = col2_vc_sorted.tail(5)

    # bar chart
    location = vc_tail.index
    count = vc_tail.values

    plt.figure(figsize=(10, 6))
    plt.bar(location, count)
    plt.title('Least Common Locations')
    plt.xlabel('Location')
    plt.ylabel('Count')

    minimumY = min(count * 0.50)
    plt.ylim(bottom=minimumY)
    plt.tight_layout()
    plt.show()


# todo write analysis comments
def task2(df):
    t2_df = df.copy(deep=True)
    result_df = pd.DataFrame({'D': [], 'result': []})

    for D in range(1, 12):
        # ensure that results are split into ranges, using &
        greater_than_D = t2_df[(abs(t2_df['Pressure9am'] - t2_df['Pressure3pm']) >= D) & (
                abs(t2_df['Pressure9am'] - t2_df['Pressure3pm']) < D + 1)]
        vc_greater_than_D = greater_than_D['RainTomorrow'].value_counts()
        result = vc_greater_than_D['Yes'] / vc_greater_than_D['No']
        result_df.loc[len(result_df) + 1] = {'D': D, 'result': result}

    # slightly different logic for last line, as I don't want to cut off differences greater than 13
    greater_than_D = t2_df[(abs(t2_df['Pressure9am'] - t2_df['Pressure3pm']) >= 12)]
    vc_greater_than_D = greater_than_D['RainTomorrow'].value_counts()
    result = vc_greater_than_D['Yes'] / vc_greater_than_D['No']
    result_df.loc[len(result_df) + 1] = {'D': 12, 'result': result}
    # print (result_df)

    # plot
    index = result_df['D']
    value = result_df['result']

    plt.figure(figsize=(10, 7))
    plt.plot(index, value)
    plt.title('D vs Rainy Days/Non Rainy Days')
    plt.xlabel('(D) The minimum difference between the pressures recorded at 9am and 3pm')
    plt.ylabel('Number of rainy days divided by number of non rainy days')

    minimumY = min(value * 0.8)
    maximumY = max(value * 1.2)
    minimumX = min(index - 1)
    maximumX = max(index + 1)
    plt.ylim(bottom=minimumY)
    plt.ylim(top=maximumY)
    plt.xlim(left=minimumX)
    plt.xlim(right=maximumX)
    plt.tight_layout()
    plt.show()


def task3(df):
    t3_df = df.copy(deep=True)

    sub_df = t3_df[[
        'WindSpeed9am',
        'WindSpeed3pm',
        'Humidity9am',
        'Humidity3pm',
        'Pressure9am',
        'Temp9am',
        'Temp3pm',
        'RainTomorrow'
    ]]

    X = sub_df.drop('RainTomorrow', axis=1)

    means = X.mean(skipna=True)
    X = X.fillna(means)

    y = sub_df['RainTomorrow']
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    feature_names = X.columns.tolist()
    columns = ['Max Depth'] + feature_names
    result_df = pd.DataFrame(columns=columns)

    # loop to run through all max depths
    upper_range = 36
    for i in range(1, upper_range):
        model = DecisionTreeClassifier(random_state=42, max_depth=i)
        model.fit(X, y_encoded)
        row = [i] + list(model.feature_importances_)
        result_df.loc[len(result_df) + 1] = row
        progressbar(i, upper_range)

    # create line plot
    plt.figure(figsize=(10, 6))

    for feature in result_df.columns.drop('Max Depth'):
        plt.plot(result_df['Max Depth'], result_df[feature], label=feature)

    plt.xlabel('Max Depth')
    plt.ylabel('Feature Importance')
    plt.title('Feature Importances vs Max Depth of Decision Tree')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Line plot generated")

    result_df.to_csv('test.csv', index=False)

    # When running the above, we get a graph which maps the importance of features against the Max depth of the tree.
    # This graph shows clearly that the "Humidity3pm" feature is by far the most important in predicting whether
    # it will rain on the next day. At a max depth of one, we can see that it is the only feature that matters,
    # with a value of 1, which makes sense as at this depth the decision tree is only making one decision.
    #
    # As the max depth increases, we can see the importance of the humidity at 3pm tapers down, as more decisions are
    # made and other features are considered, until it finally settles at approximately 30% importance. Still the most
    # important feature by over 10%


def task4(df):
    t4_df = df.copy(deep=True)
    sub_df = t4_df[[
        'WindDir9am',
        'WindDir3pm',
        'Pressure9am',
        'Pressure3pm',
        'RainTomorrow'
    ]]

    X_sub_df = sub_df.drop('RainTomorrow', axis=1)

    X_pressure = X_sub_df[[
        'Pressure9am',
        'Pressure3pm'
    ]]

    X_wind_direction = X_sub_df[[
        'WindDir9am',
        'WindDir3pm',
    ]]

    encoder = OneHotEncoder(sparse=False)
    X_wind_direction_encoded = encoder.fit_transform(X_wind_direction)

    X_dict = {"Pressure": X_pressure, "Wind Direction": X_wind_direction_encoded}

    y = sub_df['RainTomorrow']
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    for X_key, X_value in X_dict.items():
        X_train, X_test, y_train, y_test = train_test_split(X_value, y_encoded, test_size=0.333, random_state=42)

        model = DecisionTreeClassifier(random_state=42, max_depth=20)
        model.fit(X_train, y_train)

        print("\n", X_key)
        print("Training Accuracy:", f"{model.score(X_train, y_train):.5f}")
        print("Test Accuracy:", f"{model.score(X_test, y_test):.5f}")

    # Running this function provides the following results:

    # Pressure:
    # Training Accuracy: 0.77803
    # Test Accuracy: 0.75241
    #
    # Wind Direction:
    # Training Accuracy: 0.75991
    # Test Accuracy: 0.75536

    # Based on the above, we can see that on training data, The model based on Pressure more accurately identifies seen
    # data. That said, the model based on wind direction is slightly more accurate on unseen data. I would conclude from
    # this that the pressure based model may be slightly more over-fitted than the Wind Direction Model.

    # With regard to climate data, which has a wide variety of combinations I would lean towards making use of the Wind
    # direction model to predict whether it will rain tomorrow, as it is slightly more capable with unseen data,


def task5(results_df):
    t5_df = results_df.copy(deep=True)
    sub_df = t5_df[[
        'RainTomorrow',
        'WindDir9am',
        'WindGustDir',
        'WindDir3pm'
    ]]

    sub_df_clean = sub_df.dropna()

    sub_df_max_len2 = sub_df_clean[
        (sub_df_clean['WindDir9am'].apply(len) < 3) &
        (sub_df_clean['WindDir3pm'].apply(len) < 3) &
        (sub_df_clean['WindGustDir'].apply(len) < 3)
        ]

    X = sub_df_max_len2.drop('RainTomorrow', axis=1)
    y = sub_df_max_len2['RainTomorrow']

    one_hot_encoder = OneHotEncoder()
    X_encoded = one_hot_encoder.fit_transform(X)

    # svd = TruncatedSVD(n_components=24) # processing is very slow, reducing dimensionality to try to speed up.
    # X_reduced = svd.fit_transform(X_encoded)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    depth_list = []
    dtc_train_results = []
    dtc_test_results = []
    knc_train_results = []
    knc_test_results = []

    upper_range = 11
    progressbar(0, upper_range * 2 - 1)

    for i in range(1, upper_range):
        depth_list.append(i)

        dtc = DecisionTreeClassifier(random_state=42, max_depth=i)
        knc = KNeighborsClassifier(n_neighbors=i, n_jobs=-1, algorithm='brute')

        dtc_scores = cross_validate(dtc, X_encoded, y_encoded, cv=5, return_train_score=True)
        dtc_train_results.append(np.mean(dtc_scores['train_score']))
        dtc_test_results.append(np.mean(dtc_scores['test_score']))
        progressbar(i * 2 - 1, upper_range * 2 - 1)

        knc_scores = cross_validate(knc, X_encoded, y_encoded, cv=5, return_train_score=True,
                                    n_jobs=-1)  # research, trying to speed up processing https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        knc_train_results.append(np.mean(knc_scores['train_score']))
        knc_test_results.append(np.mean(knc_scores['test_score']))
        progressbar(i * 2, upper_range * 2 - 1)

    plt.figure(figsize=(10, 8))

    # Plot for Decision Tree Classifier
    plt.subplot(2, 1, 1)
    plt.plot(depth_list, dtc_train_results, label='Decision Tree Training Accuracy')
    plt.plot(depth_list, dtc_test_results, label='Decision Tree Test Accuracy')
    plt.title('Tree Classifier Accuracy')
    plt.xlabel('Parameter Value')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot for KNeighbors Classifier
    plt.subplot(2, 1, 2)
    plt.plot(depth_list, knc_train_results, label='KNeighbors Training Accuracy')
    plt.plot(depth_list, knc_test_results, label='KNeighbors Test Accuracy')
    plt.title('Classifier Accuracy')
    plt.xlabel('Parameter Value')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def task6(df):
    t6_df = df.copy(deep=True)
    sub_df = t6_df[[
        'MinTemp',
        'MaxTemp',
        'WindSpeed9am',
        'WindSpeed3pm',
        'Humidity9am',
        'Humidity3pm',
        'Pressure9am',
        'Pressure3pm',
        'Rainfall',
        'Temp9am',
        'Temp3pm'
    ]]

    cleaned_df = sub_df.dropna()

    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(cleaned_df)

    lowerRange = 2
    upperRange = 9
    kRange = range(lowerRange, upperRange)

    progressbar(0, upperRange - lowerRange)

    distortionResults = []
    silhouetteResults = []

    for i in kRange:
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(scaled_df)

        distortionResults.append(kmeans.inertia_)
        silhouetteResults.append(silhouette_score(scaled_df, kmeans.labels_))
        progressbar(i - lowerRange, upperRange - lowerRange)

    # elbow method
    plt.figure(figsize=(10, 8))
    plt.plot(kRange, distortionResults, 'bx-')
    plt.xlabel('Clusters')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method')
    plt.show()

    # doesn't give a very clear shape.
    # It's hard to estimate based on this method, but it might be around 6 clusters.
    # Will try the silhouette method.

    plt.plot(range(2, 11), silhouetteResults)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.show()


def task7(df):
    t7_df = df.copy(deep=True)


def progressbar(i, upper_range):
    # Some functions in this project take some time to run due to loops.
    # This gives visual indication of progress
    progress_string = f'\r{("#" * i)}{("_" * ((upper_range - 1) - i))} {i} / {upper_range - 1}'
    if i == upper_range - 1:
        print(progress_string)
    else:
        print(progress_string, end='')


if __name__ == '__main__':
    main()
