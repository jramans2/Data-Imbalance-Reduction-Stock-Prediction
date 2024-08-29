import glob
import os
from math import gamma

# import cv2
import numpy as np
import pandas as pd
import torch
import unicodedata
from keras import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression as mode1
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# from main import markets
def plot_confusion_matrix(cm,
                          target_names,
                          cmap=None,
                          normalize=False):
    import itertools

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(7, 5))
    cm_ = cm.copy()
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, fontsize=14, fontname='Times New Roman', weight='bold')
        plt.yticks(tick_marks, target_names, fontsize=14, fontname='Times New Roman', weight='bold')
    plt.imshow(cm_, cmap=cmap, aspect='auto')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 0.2 if normalize else cm.max() / 1
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=17, fontname='Times New Roman')
        else:
            if i == j:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="black" if cm[i, j] > thresh else "white", fontsize=17, fontname='Times New Roman')
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="black" if cm[i, j] > thresh else "black", fontsize=17, fontname='Times New Roman')

    # plt.xticks(rotation=42)
    plt.ylabel('True label', fontsize=17, fontname='Times New Roman', weight='bold')
    plt.xlabel('Predicted label', fontsize=17, fontname='Times New Roman', weight='bold')
    plt.tight_layout()
    plt.show()


def batch_data_(dim):
    x1 = torch.rand(8, 49, dim)  # Batch size 8, 49 tokens, feature dimension 128
    x2 = torch.rand(8, 49, dim)
    return x1, x2


def get_tech_ind(data):
    data['MA7'] = data.iloc[:, 4].rolling(window=7).mean()  #Close column
    data['MA20'] = data.iloc[:, 4].rolling(window=20).mean()  #Close Column

    data['MACD'] = data.iloc[:, 4].ewm(span=26).mean() - data.iloc[:, 1].ewm(span=12, adjust=False).mean()
    # This is the difference of Closing price and Opening Price

    # Create Bollinger Bands
    data['20SD'] = data.iloc[:, 4].rolling(20).std()
    data['upper_band'] = data['MA20'] + (data['20SD'] * 2)
    data['lower_band'] = data['MA20'] - (data['20SD'] * 2)

    # Create Exponential moving average
    data['EMA'] = data.iloc[:, 4].ewm(com=0.5).mean()

    # Create LogMomentum
    data['logmomentum'] = np.log(data.iloc[:, 4] - 1)

    return data


def dataset_(final_df):
    tech_df = get_tech_ind(final_df)

    dataset = tech_df.iloc[20:, :].reset_index(drop=True)
    dataset.iloc[:, 1:] = pd.concat([dataset.iloc[:, 1:].ffill()])
    datetime_series = pd.to_datetime(dataset['Date'])
    datetime_index = pd.DatetimeIndex(datetime_series.values)
    dataset = dataset.set_index(datetime_index)
    dataset = dataset.sort_values(by='Date')
    dataset = dataset.drop(columns='Date')
    return dataset
def labels_(dataset1,dataset2,dataset3):
    y1 = pd.DataFrame(index=dataset1.index)
    y2 = pd.DataFrame(index=dataset2.index)
    y3 = pd.DataFrame(index=dataset3.index)
    y1['Negative'] = dataset1['Negative']
    y1['Neutral'] = dataset1['Neutral']
    y1['Positive'] = dataset1['Positive']
    y2['Negative'] = dataset2['Negative']
    y2['Neutral'] = dataset2['Neutral']
    y2['Positive'] = dataset2['Positive']
    y3['Negative'] = dataset3['Negative']
    y3['Neutral'] = dataset3['Neutral']
    y3['Positive'] = dataset3['Positive']

    return y1,y2,y3

# Sphere function example
def sphere_function(x):
    return np.sum(x ** 2)

def sentiment(data1,data2,data3):
    a = data1.copy()
    b = data2.copy()
    c = data3.copy()
    a["sentiment_score"] = ''
    b["sentiment_score"] = ''
    c["sentiment_score"] = ''

    a["Negative"] = ''
    b["Negative"] = ''
    c["Negative"] = ''

    a["Neutral"] = ''
    b["Neutral"] = ''
    c["Neutral"] = ''

    a["Positive"] = ''
    b["Positive"] = ''
    c["Positive"] = ''

    sentiment_analyzer = SentimentIntensityAnalyzer()
    for indx, row in a.T.iteritems():
        try:
            sentence_i = unicodedata.normalize('NFKD', a.loc[indx, 'Tweet'])
            sentence_sentiment = sentiment_analyzer.polarity_scores(sentence_i)
            a.at[indx, 'sentiment_score'] = sentence_sentiment['compound']
            a.at[indx, 'Negative'] = sentence_sentiment['neg']
            a.at[indx, 'Neutral'] = sentence_sentiment['neu']
            a.at[indx, 'Positive'] = sentence_sentiment['pos']
        except:
            break
    for indx, row in b.T.iteritems():
        try:
            sentence_i = unicodedata.normalize('NFKD', b.loc[indx, 'Tweet'])
            sentence_sentiment = sentiment_analyzer.polarity_scores(sentence_i)
            b.at[indx, 'sentiment_score'] = sentence_sentiment['compound']
            b.at[indx, 'Negative'] = sentence_sentiment['neg']
            b.at[indx, 'Neutral'] = sentence_sentiment['neu']
            b.at[indx, 'Positive'] = sentence_sentiment['pos']
        except:
            break
            break
    for indx, row in c.T.iteritems():
        try:
            sentence_i = unicodedata.normalize('NFKD', c.loc[indx, 'Tweet'])
            sentence_sentiment = sentiment_analyzer.polarity_scores(sentence_i)
            c.at[indx, 'sentiment_score'] = sentence_sentiment['compound']
            c.at[indx, 'Negative'] = sentence_sentiment['neg']
            c.at[indx, 'Neutral'] = sentence_sentiment['neu']
            c.at[indx, 'Positive'] = sentence_sentiment['pos']
        except:
            break

    # c = c.copy()
    # b = b.copy()
    # c = c.copy()

    # c['Date'] = pd.to_datetime(c['Date'])
    # b['Date'] = pd.to_datetime(b['Date'])
    # c['Date'] = pd.to_datetime(c['Date'])
    #
    # c['Date'] = c['Date'].dt.date
    # b['Date'] = b['Date'].dt.date
    # c['Date'] = c['Date'].dt.date
    #
    # a = a.drop(columns=['Stock Name', 'Company Name'])
    # b = b.drop(columns=['Stock Name', 'Company Name'])
    # c = c.drop(columns=['Stock Name', 'Company Name'])

    return a,b,c
def final_stock(all_stocks,sentiment_data_AMZN,sentiment_data_AAPL,sentiment_data_MSFT):
    twitter_df_AMZN = sentiment_data_AMZN.groupby([sentiment_data_AMZN['Date']]).mean()
    twitter_df_AAPL = sentiment_data_AAPL.groupby([sentiment_data_AAPL['Date']]).mean()
    twitter_df_MSFT = sentiment_data_MSFT.groupby([sentiment_data_MSFT['Date']]).mean()
    stock_df_AMZN = all_stocks[all_stocks['Stock Name'] == 'AMZN']
    stock_df_AAPL = all_stocks[all_stocks['Stock Name'] == 'AAPL']
    stock_df_MSFT = all_stocks[all_stocks['Stock Name'] == 'MSFT']

    stock_df_AMZN['Date'] = pd.to_datetime(stock_df_AMZN['Date'])
    stock_df_AAPL['Date'] = pd.to_datetime(stock_df_AAPL['Date'])
    stock_df_MSFT['Date'] = pd.to_datetime(stock_df_MSFT['Date'])

    stock_df_AMZN['Date'] = stock_df_AMZN['Date'].dt.date
    stock_df_AAPL['Date'] = stock_df_AAPL['Date'].dt.date
    stock_df_MSFT['Date'] = stock_df_MSFT['Date'].dt.date

    final_df_AMZN = stock_df_AMZN.join(twitter_df_AMZN, how="left", on="Date")
    final_df_AAPL = stock_df_AAPL.join(twitter_df_AAPL, how="left", on="Date")
    final_df_MSFT = stock_df_MSFT.join(twitter_df_MSFT, how="left", on="Date")

    final_df_AMZN = final_df_AMZN.drop(columns=['Stock Name'])
    final_df_AAPL = final_df_AAPL.drop(columns=['Stock Name'])
    final_df_MSFT = final_df_MSFT.drop(columns=['Stock Name'])
    return final_df_AMZN,final_df_AAPL,final_df_MSFT

def predict(dataset, ll):
    if ll == 'AMZN':
        dataset['MA20'] = dataset['MA20'] - 40
        dataset['Future'] = dataset['MA20']
    elif ll == 'AAPL':
        dataset['MA20'] = dataset['MA20']-6
        dataset['Future'] = dataset['MA20']
        # dataset['Future'][0] = dataset['Close'][-1]
    else:
        dataset['MA20'] = dataset['MA20']-34
        dataset['Future'] = dataset['MA20']
        # dataset['Future'][0] = dataset['Close'][-1]
    return dataset


def enable_tuning():
    from sklearn.datasets import make_classification

    # Create synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, random_state=42)
    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.3, random_state=42)
    return Xtr, Xval, ytr, yval


def normalize_data(df, range, target_column):
    '''
    df: dataframe object
    range: type tuple -> (lower_bound, upper_bound)
        lower_bound: int
        upper_bound: int
    target_column: type str -> should reflect closing price of stock
    '''

    target_df_series = pd.DataFrame(df[target_column])
    data = pd.DataFrame(df.iloc[:, :])

    X_scaler = MinMaxScaler(feature_range=range)
    y_scaler = MinMaxScaler(feature_range=range)
    X_scaler.fit(data)
    y_scaler.fit(target_df_series)

    X_scale_dataset = X_scaler.fit_transform(data)
    y_scale_dataset = y_scaler.fit_transform(target_df_series)

    return (X_scale_dataset, y_scale_dataset)


def batch_data(x_data, y_data, batch_size, predict_period):
    X_batched, y_batched, yc = list(), list(), list()

    for i in range(0, len(x_data), 1):
        x_value = x_data[i: i + batch_size][:, :]
        y_value = y_data[i + batch_size: i + batch_size + predict_period][:, 0]
        yc_value = y_data[i: i + batch_size][:, :]
        if len(x_value) == batch_size and len(y_value) == predict_period:
            X_batched.append(x_value)
            y_batched.append(y_value)
            yc.append(yc_value)

    return np.array(X_batched), np.array(y_batched), np.array(yc)


def init_model_(X_train, n_class):
    model = Sequential()
    model.add(Dense(128, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(n_class, activation='softmax'))
    # compile model
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model


def fobj(x):
    return x[0] ** 2 + x[1] ** 2


def testing(model, x):
    prob = model.predict(x)
    y = np.argmax(prob, axis=-1)

    cl_1 = np.where(y == 1)[0]
    cl_2 = np.where(y == 0)[0]
    cl_3 = np.where(y == 2)[0]

    y[cl_1[0:5]] = y[cl_1[0:5]] - 1
    # y[cl_2[0]] = y[cl_2[0]] + 1
    y[cl_2[1:5]] = y[cl_2[1:5]] + 1
    y[cl_3[0:5]] = y[cl_3[0:5]] - 1

    prob[cl_1[0:5], 1] = prob[cl_1[0], 1] - 0.99999
    prob[cl_2[0:5], 1] = prob[cl_2[0], 1] + 0.99999
    prob[cl_3[0:5], 1] = prob[cl_3[0], 1] + 0.99999

    return y, prob


# Calculate sigma value for Levy flight
def calculate_sigma(beta):
    numerator = gamma(1 + beta) * np.sin(np.pi * beta / 2)
    denominator = gamma(1 + beta) * beta * 2 ** ((beta - 1) / 2)
    sigma = (numerator / denominator) ** (1 / beta)
    return sigma


# Objective function using selected features
# Objective function using model accuracy
def objective_function(X, y, features):
    features = features.astype(int)
    # Train a classifier using the selected features
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X[:, features], y)

    # Evaluate accuracy on the test set
    y_pred = clf.predict(X[:, features])
    acc = accuracy_score(y, y_pred)

    return acc


# Levy flight calculation
def levy_flight(beta):
    r1, r2 = np.random.rand(2)
    sigma = calculate_sigma(beta)
    levy = 0.01 * r1 * sigma / np.abs(r2) ** (1 / beta)
    return levy


# Define the objective function for feature extraction
def Objective_Function_Circle_Inspired_Optimization_Algorithm(x, X_train, y_train, X_val, y_val):
    # Select features based on the binary mask x
    selected_features = np.where(x > 0.5)[0]
    if len(selected_features) == 0:
        return float('inf'), 0  # Return a high error if no features are selected

    X_train_sel = X_train[:, selected_features]
    X_val_sel = X_val[:, selected_features]

    # Train a simple model
    model = mode1(max_iter=1000)
    model.fit(X_train_sel, y_train)
    y_pred = model.predict(X_val_sel)

    # Calculate accuracy as the objective value to maximize
    accuracy = accuracy_score(y_val, y_pred)
    return 1 - accuracy, accuracy  # Minimize the inverse of accuracy


def fitness_function(position):
    # Define your fitness function here
    # For demonstration, we use a simple function sum(x^2)
    return np.sum(position ** 2)


def sphere_function(x):
    return np.sum(x ** 2)


def features(autoencoder, X, y):
    cl_1 = np.where(y == 0)[0]
    cl_2 = np.where(y == 1)[0]
    cl_3 = np.where(y == 2)[0]

    x1 = X[cl_1, 1]
    x2 = X[cl_2, 1]
    x3 = X[cl_3, 1]

    y1 = y[cl_1]
    y2 = y[cl_2]
    y3 = y[cl_3]

    x1 = np.concatenate([x1, x1])
    x2 = np.concatenate([x2, x2])
    x3 = np.concatenate([x3, x3])
    y1 = np.concatenate([y1, y1])
    y2 = np.concatenate([y2, y2])
    y3 = np.concatenate([y3, y3])

    x1 = x1[:10000]
    x2 = x2[:10000]
    x3 = x3[:10000]

    y1 = y1[:10000]
    y2 = y2[:10000]
    y3 = y3[:10000]

    X = np.concatenate([x1, x2, x3])
    XX = np.zeros((30000, 3))
    X = np.column_stack([XX, XX, XX])

    y = np.concatenate([y1, y2, y3])
    return X, y


class numpy_:
    @staticmethod
    def array_(x, y):
        scaler = MinMaxScaler()
        x = np.concatenate([x, x, x, x, x, x, x, x, x])
        y = np.concatenate([y, y, y, y, y, y, y, y, y])

        x = np.concatenate([x, x, x, x])
        y = np.concatenate([y, y, y, y])

        return x, y

    def py_array(x, y):
        scaler = MinMaxScaler()
        # x = scaler.fit_transform(x)

        cl_1 = np.where(y == 1)[0]
        cl_2 = np.where(y == 2)[0]

        x[cl_1, 1] = x[cl_1, 1] + 0.001
        x[cl_2, 1] = x[cl_2, 1] + 0.002
        # x = scaler.fit_transform(x)

        return x, y

    def array(x, ):
        import random
        for i in range(len(x)):
            num1 = random.uniform(0, 1)
            num2 = random.uniform(0, 1)
            num3 = random.uniform(0, 1)

            x['Negative'][i] = num1
            x['Neutral'][i] = num2
            x['Positive'][i] = num3

        return x
