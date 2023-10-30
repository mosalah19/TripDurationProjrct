import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split


def load_train_dataset(url_train=r"D:\Data Science\course_ML_mostafa_saad\projects\project-taxi-trip-duration\datasets\train.csv"):
    df = pd.read_csv(rf"{url_train}")
    mask = np.random.rand(len(df)) <= 0.99
    return df[mask], df[~mask], df


def load_val_dataset(url_validation=r"D:\Data Science\course_ML_mostafa_saad\projects\project-taxi-trip-duration\datasets\val.csv"):
    df = pd.read_csv(rf"{url_validation}")
    return df


def split_data(df):
    # separeted target from dataframe
    X_train = df.drop('trip_duration', axis=1)
    y_train = df['trip_duration']

    return X_train, y_train


def preprocessing_scalling(df, choice=1):
    if choice == 0:
        return df, np.null
    elif choice == 1:
        processor = MinMaxScaler()
        return processor.fit_transform(df), processor
    else:
        processor = StandardScaler()
        return processor.fit_transform(df), processor


def monomials(train_data, degree=1):
    if degree > 1:
        new_arr_t = []
        for q in range(train_data.shape[1]):
            new_arr_t.extend([np.array((train_data.iloc[:, q] ** s))
                              for s in range(2, degree+1)])

        x1 = np.hstack((train_data, *(i.reshape(-1, 1) for i in new_arr_t)))
        return x1
    else:
        return train_data


def transformation(df, choice=1, degree=1):
    if choice == 1:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        df = poly.fit_transform(df)
        return df
    if choice == 2:
        df = monomials(train_data=df, degree=degree)
        return df
