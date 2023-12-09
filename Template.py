#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on a cloudy day


Name: Emmett Fitzharris
Student ID: R00222357
Cohort: evSD3

"""

import pandas as pd
import numpy as np
import sklearn as skl
import matplotlib as mpl

# Use the below few lines to ignore the warning messages

import warnings

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

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
    task4(df)
    # task5(df)
    # task6(df)
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

    X_list = [X_pressure, X_wind_direction_encoded]

    y = sub_df['RainTomorrow']
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    for X in X_list:
        X_train, X_test, y_train, y_test  = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        model = DecisionTreeClassifier(random_state=42, max_depth=20)
        model.fit(X_train, y_train)

        print(X)
        print("Training Accuracy:", model.score(X_train, y_train))
        print("Test Accuracy:", model.score(X_test, y_test))



def task5(df):
    t5_df = df.copy(deep=True)
    sub_df = t5_df[[
        'RainTomorrow',
        'Wind-Dir9am',
        'WindGustDir',
        'WindDir3pm'
    ]]


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


def task7(df):
    t7_df = df.copy(deep=True)


def progressbar(i, upper_range):
    # Some functions in this project take some time to run due to loops.
    # This gives visual indication of progress
    progress_string = f'\r{("#" * i)}{("_" * (upper_range - i))} {i} / {upper_range - 1}'
    if i == upper_range - 1:
        print(progress_string)
    else:
        print(progress_string, end='')


if __name__ == '__main__':
    main()
