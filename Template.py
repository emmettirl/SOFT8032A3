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

warnings.filterwarnings("ignore", category=DeprecationWarning)
def warn(*args, **kwargs):
    pass


def main():
    warnings.warn = warn

    df = pd.read_csv("weather.csv")
    # print(df.head())

    # task1(df)
    task2(df)
    # task3(df)
    # task4(df)
    # task5(df)
    # task6(df)
    # task7(df)


def  task1(df):
    t1_df = df.copy(deep=True)

    #column 2
    col2 = df.iloc[:, 1]

    #data cleaning
    col2_clean = col2.str.lower()

    #print number of unique locations
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


def task2(df):
    t2_df = df.copy(deep=True)
    result_df = pd.DataFrame({'D': [], 'result': []})


    for D in range (1, 12):
        # ensure that results are split into ranges, using &
        greater_than_D = t2_df[(abs(t2_df['Pressure9am'] - t2_df['Pressure3pm']) >= D) & (abs(t2_df['Pressure9am'] - t2_df['Pressure3pm']) < D+1)]
        vc_greater_than_D = greater_than_D['RainTomorrow'].value_counts()
        result = vc_greater_than_D['Yes'] / vc_greater_than_D['No']
        result_df.loc[len(result_df)+1] = {'D': D, 'result': result}

    # slightly different logic for last line, as I don't want to cut off differences greater than 13
    greater_than_D = t2_df[(abs(t2_df['Pressure9am'] - t2_df['Pressure3pm']) >= 12)]
    vc_greater_than_D = greater_than_D['RainTomorrow'].value_counts()
    result = vc_greater_than_D['Yes'] / vc_greater_than_D['No']
    result_df.loc[len(result_df)+1] = {'D': 12, 'result': result}
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


        
def task4(df):
    t4_df = df.copy(deep=True)



def task5(df):
    t5_df = df.copy(deep=True)



def task6(df):
    t6_df = df.copy(deep=True)



def task7(df):
    t7_df = df.copy(deep=True)



if __name__ == '__main__':
    main()