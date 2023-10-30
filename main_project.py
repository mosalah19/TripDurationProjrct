from data_helper import *
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import argparse
import pickle


def remove_unnecessary_columns(df):
    # dropoff_datetime used only for compute duration
    df = df.drop(['id'], axis=1)
    return df


def feature_Engineering_date_time(df):
    # transform column from object to datetime
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    # extract column from it
    df['month'] = df['pickup_datetime'].dt.month.astype('int64')
    df['day'] = df['pickup_datetime'].dt.day.astype('int64')
    df['hour'] = df['pickup_datetime'].dt.hour.astype('int64')
    df['dayofyear'] = df['pickup_datetime'].dt.dayofyear.astype('int64')
    df['dayofweek'] = df['pickup_datetime'].dt.dayofweek.astype('int64')
    # make column contain if trip in weekend or not
    df['is_weekend'] = (df['dayofweek'] >= 5).astype('int64')
    # divide day to sessions
    b = [0, 4, 8, 12, 16, 20, 24]
    l = ['Late Night', 'Early Morning', 'Morning', 'Noon', 'Eve', 'Night']
    df['session'] = pd.cut(df['hour'], bins=b, labels=l, include_lowest=True)
    # drop date column
    df = df.drop('pickup_datetime', axis=1)
    return df


def feature_Engineering_categorical_data(df):
    # replace Y ->1 , N->0
    df['store_and_fwd_flag'] = (
        df['store_and_fwd_flag'] == 'Y').astype('int64')
    df = apply_one_hot_encoding(df, ['session'])
    return df


def apply_one_hot_encoding(df, column_name):
    # creat object from one hot encoding
    # The sparse=False argument outputs a non-sparse matrix.
    df[column_name].astype('object')
    ohe = OneHotEncoder(sparse_output=False)
    array = ohe.fit_transform(df[column_name])
    categoricals = np.array(ohe.categories_).ravel()
    x = pd.DataFrame(
        array, columns=[f'{column_name[0]}_{ca}' for ca in categoricals])
    df.index = range(df.shape[0])
    df = pd.concat([df, x], axis=1)
    df = df.drop(column_name, axis=1)
    return df


def calculate_distance(lat1, lon1, lat2, lon2):
    # Check if the coordinates are the same
    same_location = (lat1 == lat2) & (lon1 == lon2)
    # Set distance to 0 for same locations
    distance = same_location.astype(float) * 0.0

    # Convert degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # Calculate the differences between the coordinates
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = np.sin(dlat/2) ** 2 + np.cos(lat1_rad) * \
        np.cos(lat2_rad) * np.sin(dlon/2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    # Earth's radius in miles
    radius = 3959

    # Calculate the distance for non-same locations
    non_same_locations = ~same_location
    distance[non_same_locations] = radius * c[non_same_locations]

    return distance


def remove_outliers_from_trip_duration_and_distance(df):
    df = df.drop(df[df['trip_duration'] >= (22*60*60)].index)
    df = df.drop(df[df['trip_duration'] <= (2*60)].index)
    outliers = df['distance'].quantile(0.99)
    df = df[df['distance'] <= outliers]
    return df


def addExtraFeatures_weather(df, url=r"D:\Data Science\course_ML_mostafa_saad\projects\project-taxi-trip-duration\datasets\weather_data_nyc_centralpark_2016(1).csv"):
    wether = pd.read_csv(rf"{url}")

    wether['snow depth'].replace("T", "1", inplace=True)
    wether['snow fall'].replace("T", "1", inplace=True)
    wether['precipitation'].replace("T", "1", inplace=True)
    wether['snow depth'] = wether['snow depth'].astype('float64')
    wether['snow fall'] = wether['snow fall'].astype('float64')
    wether['precipitation'] = wether['precipitation'].astype('float64')
    wether.drop('date', axis=1, inplace=True)
    x = pd.merge(df, wether, on='dayofyear')
    return x


def addExtraFeatures_osrm(df,
                          url=[r"D:\Data Science\course_ML_mostafa_saad\projects\project-taxi-trip-duration\datasets\fastest_routes_train_part_1.csv",
                               r"D:\Data Science\course_ML_mostafa_saad\projects\project-taxi-trip-duration\datasets\fastest_routes_train_part_2.csv", r"D:\Data Science\course_ML_mostafa_saad\projects\project-taxi-trip-duration\datasets\fastest_routes_test.csv"]):

    osrm_p1 = pd.read_csv(rf"{url[0]}")
    osrm_p2 = pd.read_csv(rf"{url[1]}")
    osrm_p3 = pd.read_csv(rf"{url[2]}")
    osrm_p1 = osrm_p1[['id', 'total_distance',
                       'total_travel_time', 'number_of_steps']]
    osrm_p2 = osrm_p2[['id', 'total_distance',
                       'total_travel_time', 'number_of_steps']]
    osrm_p3 = osrm_p3[['id', 'total_distance',
                       'total_travel_time', 'number_of_steps']]
    osrm = pd.concat([osrm_p1, osrm_p2, osrm_p3], ignore_index=True)
    df = pd.merge(df, osrm, on='id', how='left')
    df.dropna(inplace=True)
    df['total_distance'] = np.log1p(df['total_distance'])
    df['total_travel_time'] = np.log1p(df['total_travel_time'])
    df['number_of_steps'] = np.log1p(df['number_of_steps'])

    return df


def feature_Engineering_continuous_data(df):
    # from EDA can you a trips thay has 7 or zero number of passanger is very few
    # i decide to remove it
    indexAge = df[(df['passenger_count'] >= 7) |
                  (df['passenger_count'] == 0)].index
    df.drop(indexAge, inplace=True)
    df.index = range(df.shape[0])

    # from pickup_longitude	pickup_latitude	dropoff_longitude	dropoff_latitude:
    # can extract new feature like distance
    df['distance'] = calculate_distance(
        df['pickup_latitude'].to_numpy(), df['pickup_longitude'].to_numpy(), df['dropoff_latitude'].to_numpy(), df['dropoff_longitude'].to_numpy())
    # remove outliers from distance and trip duration
    df = remove_outliers_from_trip_duration_and_distance(df)
    # some of feature have large scale like trip duration and distance
    df['trip_duration'] = np.log1p(df['trip_duration'])
    df['distance'] = np.log1p(df['distance'])

    return df


def scale_data(df_train, df_val, choice=2):
    df_train, scaler = preprocessing_scalling(
        df_train, choice=choice)
    df_val = scaler.transform(df_val)
    return df_train, df_val, scaler


def predict_eval(model, train, train_features, name, model_name):
    y_train_pred = model.predict(train)
    rmse = mean_squared_error(train_features, y_train_pred, squared=False)
    r2 = r2_score(train_features, y_train_pred)
    print(model_name+"  : ")
    print(f"{name} RMSE = {rmse:.4f} - R2 = {r2:.4f}")


def Ridge_with_polynomial(train_x, train_y, val_x='non', val_y='non', degree=2, intercipt=True):

    pipeline = Pipeline(steps=[
        ('preprocessor', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('regression', Ridge(alpha=1, fit_intercept=intercipt))
    ])

    model = pipeline.fit(train_x, train_y)

    predict_eval(model, train_x, train_y, "train",
                 model_name='Ridge_with_polynomial')
    # predict_eval(model, val_x, val_y, "test",
    #              model_name='Ridge_with_polynomial')
    return model


def simple_Ridge(train_x, train_y, val_x, val_y, intercipt=True):

    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('regression', LinearRegression(fit_intercept=intercipt))
    ])

    model = pipeline.fit(train_x, train_y)

    predict_eval(model, train_x, train_y, "train", model_name='simple_Ridge')
    predict_eval(model, val_x, val_y, "test", model_name='simple_Ridge')
    return model


def random_forest_regression(X_train, X_test, y_train, test_y):
    # Create and train the Random Forest regression model
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('regression', RandomForestRegressor(
            n_estimators=25, min_samples_leaf=25, min_samples_split=25))
    ])
    model = pipeline.fit(X_train, y_train)
    predict_eval(model, train_x, train_y, "train",
                 model_name='random_forest_regression')
    predict_eval(model, X_test, test_y, "test",
                 model_name='random_forest_regression')
    return model


def xgboost_regression(X_train, X_test, y_train, test_y):
    # Create and train the XGBoost regression model
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('regression', XGBRegressor(random_state=42))
    ])
    model = pipeline.fit(X_train, y_train)
    predict_eval(model, train_x, train_y, "train",
                 model_name='xgboost_regression')
    predict_eval(model, X_test, test_y, "test",
                 model_name='xgboost_regression')
    return model


def prepare_data(df):
    # deal with datetime
    df = feature_Engineering_date_time(df)
    # add exta feature
    df = addExtraFeatures_weather(df)
    # add osrm data
    df = addExtraFeatures_osrm(df)
    # remove unnecessary columns
    df = remove_unnecessary_columns(df)
    # deal with categorical data
    df = feature_Engineering_categorical_data(df)
    # deal with continuous data
    df = feature_Engineering_continuous_data(df)
    return df


# model = simple_Ridge(train_x, train_y, val_x, val_y)
if __name__ == '__main__':

    # load dataset
    df_train, df_test, df = load_train_dataset()
    df = df.drop(['dropoff_datetime'], axis=1)
    df_val = load_val_dataset()

    ''' 
    check model by divide dataset to trail and val 
    df_val = df_val.drop(['dropoff_datetime'], axis=1)
    # prepar train data
    df_train = prepare_data(df)
    # prepare train data
    df_val = prepare_data(df_val)
    # divide dataframe into input columns  (x) and output column
    train_x = df_train.drop('trip_duration', axis=1)
    train_y = df_train['trip_duration']
    val_x = df_val.drop('trip_duration', axis=1)
    val_y = df_val['trip_duration']
    model2 = Ridge_with_polynomial(
        train_x, train_y, val_x, val_y, degree=2, intercipt=True)
    '''
    '''
Ridge_with_polynomial  :
train RMSE = 0.3564 - R2 = 0.7298
Ridge_with_polynomial  :
test RMSE = 0.3559 - R2 = 0.7311
    '''
    # final model train on all data
    df = pd.concat([df, df_val], ignore_index=True)
    # this column used only for compute trip duration
    df = df.drop(['dropoff_datetime'], axis=1)
    # make preprocessing on data
    df = prepare_data(df)
    # split to data and target
    train_x = df.drop('trip_duration', axis=1)
    train_y = df['trip_duration']
    # train model and evaluate by rmse and r2_score
    model = Ridge_with_polynomial(
        train_x, train_y, degree=2, intercipt=True)
   # save model with pickle
    with open('tripDurationProduction.pkl', 'wb') as file:
        pickle.dump(model, file)
