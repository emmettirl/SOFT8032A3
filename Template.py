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

    task1(df)
    # task2(df)
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