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
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
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

    task1(df)
    task2(df)
    task3(df)
    task4(df)
    task5(df)
    task6(df)
    task7(df)


def task1(df):
    printDivider("Task 1")
    t1_df = df.copy(deep=True)

    # column 2
    col2 = df.iloc[:, 1]

    # data cleaning
    col2_clean = col2.str.lower()

    # print number of unique locations
    print(f' Unique locations: {col2_clean.nunique()}')

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
    print("plot shown 'Least Common Locations'")

    # this graph shows that the least common locations are norahead, salmon gums, katherine, nhil and uluru.
    # There are 49 locations in total.

def task2(df):
    printDivider("Task 2")
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
    print("plot shown: 'D vs Rainy Days/Non Rainy Days'")

    # the resulting graph shows that difference in pressure between 9am and 3pm is a good indicator of whether it will
    # rain or not. The ratio of rainy days increases as the difference in pressure does,
    # and very sharply from a differnce of 11 onwards.


def task3(df):
    printDivider("Task 3")
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
    print("plot shown: 'Feature Importances vs Max Depth of Decision Tree'")

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
    printDivider("Task 4")
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

        print(X_key)
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
    printDivider("Task 5")
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
    plt.title('KNeighbors Classifier Accuracy')
    plt.xlabel('Parameter Value')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
    print("plot shown: 'KNeighbors Classifier Accuracy'")

    # The two plots generated by this function clearly show the differences between the two models.
    # We can see with the decision tree classifier, we actually get pretty strong results with a relatively low depth.
    # About two decisions in, we see the accuracy with test data is starting to drop. This is a result of over-fitting.
    # It is very clear that we can see the training accuracy is increasing as the depth increases,
    # but the accuracy with unseen data is dropping with more depth.

    # The ideal depth seems to be about 2 for this data.based on the graph, due to the high performance on unseen data.

    # On the other hand, we can see a much closer relationship between training and test accuracy with the
    # K Neighbours method. We see an early high accuracy, and then a steep drop at 3 neighbours, which is quickly
    # corrected. There is a relatively steady trend upwards in accuracy alongside neighbours.

    # At 10 neighbours, it is not entirely clear if we are seeing the end of the trend. It may be that it continues to
    # trend upward, but from the data available in this test, I would estimate that 8 neighbours is the ideal value.


def task6(df):
    printDivider("Task 6")
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
    for column in cleaned_df.columns:
        cleaned_df[column] = scaler.fit_transform(cleaned_df[[column]])



    lowerRange = 2
    upperRange = 9
    kRange = range(lowerRange, upperRange)

    distortionResults = []
    silhouetteResults = []

    # plotting all scaled columns against the rainfall column.
    # this seems to be the most sensible comparison for the plot.
    plt.figure(figsize=(10, 8))
    for column in cleaned_df.columns:
        if column != 'Rainfall':
            plt.scatter(cleaned_df['Rainfall'], cleaned_df[column], label=column)

    plt.xlabel('Rainfall')
    plt.ylabel('Values')
    plt.title('Scatter Plot of Various Columns against Rainfall')
    plt.legend()
    plt.show()
    print("plot shown: 'Scatter Plot of Various Columns against Rainfall'")

    svd = TruncatedSVD(n_components=5) # processing is very slow, reducing dimensionality to try to speed up.
    X = svd.fit_transform(cleaned_df)

    # train model
    for i in kRange:
        kmeans = KMeans(n_clusters=i, n_init=5, max_iter=100)
        kmeans.fit(X)

        distortionResults.append(kmeans.inertia_)
        silhouetteResults.append(silhouette_score(X, kmeans.labels_))
        progressbar(i - lowerRange, upperRange - lowerRange)

    # elbow method
    plt.figure(figsize=(10, 8))
    plt.plot(kRange, distortionResults, 'bx-')
    plt.xlabel('Clusters')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method')
    plt.show()
    print("plot shown: 'The Elbow Method'")

    # doesn't give a very clear shape.
    # It's hard to estimate based on this method, but it might be around 4 clusters.

    # Silhouette method.
    plt.plot(kRange, silhouetteResults)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method')
    plt.show()
    print("plot shown: 'Silhouette Method'")

    # This method seems to show a peak at 2 clusters, which indicates that 2 clusters is the ideal number. while the
    # accuracy does begin to rise again at 4 clusters, it is not by much, and the accuracy is still lower than at 2
    # clusters. It is clear that additional clusters are not adding any value to the model, in fact they are reducing
    # the accuracy significantly.

def task7(df):
    printDivider("Task 7")
    # I will create a model which predicts the wind gust speed based on the wind speed, pressure and humidity.
    # this would be useful for predicting dangerous weather conditions, and could be used to warn people of
    # potential danger.

    # I'll try a few different models, and see which one performs best.

    t7_df = df.copy(deep=True)

    t7_df_cleaned = t7_df.dropna()


    X = t7_df_cleaned[[
        'WindSpeed9am',
        'WindSpeed3pm',
        'Humidity9am',
        'Humidity3pm',
        'Pressure9am',
        'Pressure3pm'
    ]]
    y = t7_df_cleaned['WindGustSpeed']

    model_list = [DecisionTreeClassifier(), LogisticRegression(), KNeighborsClassifier(), RandomForestClassifier()]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333, random_state=42)

    runModelList(model_list, X_train, y_train, X_test, y_test, X, y)


    # The best model is the random forest classifier, though all seem to have very similar low accuracy.
    # This is likely due to the fact that the data is not very well correlated.

    # The data used is not very useful for predicting wind gust speed. I'll rerun the experiment with fewer features to see
    # if that improves performance, as pressure and humidity may have less impact than I theorized. .

    new_X = t7_df_cleaned[[
        'WindSpeed9am',
        'WindSpeed3pm',
    ]]

    X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.333, random_state=42)

    runModelList(model_list, X_train, y_train, X_test, y_test, X, y)

    # The accuracy is still very low, and the random forest classifier is still the best model.
    # it is possible that the data is not very useful for predicting wind gust speed, and that the data is not very
    # well correlated.

def runModelList(model_list, X_train, y_train, X_test, y_test, X, y):
    # Initialize the model
    for model in model_list:
        # Cross-validation
        scores = cross_val_score(model, X, y, cv=5)
        print(f"Cross-validation scores for model {type(model)}: {scores.mean():.2f} ± {scores.std():.2f}")

        model.fit(X_train, y_train)

        print("Training Accuracy:", f"{model.score(X_train, y_train):.5f}")
        print("Test Accuracy:", f"{model.score(X_test, y_test):.5f}")




def progressbar(i, upper_range):
    # Some functions in this project take some time to run due to loops.
    # This gives visual indication of progress
    progress_string = f'\r{("#" * i)}{("_" * ((upper_range - 1) - i))} {i} / {upper_range - 1}'
    if i == upper_range - 1:
        print(progress_string)
    else:
        print(progress_string, end='')

def printDivider(title):
    i = (100 - (len(title)+2))/2
    titleString = f'{"#" * int(i)} {title} {"#" * int(i)}'

    print()
    print('-' * 100)
    print(titleString)
    print('-' * 100)
    print()

if __name__ == '__main__':
    main()
