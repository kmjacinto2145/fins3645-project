#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Imports libraries for data manipulation and visualisation for the portfolio management section
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import scipy.optimize as sco
import scipy.interpolate as sci

##INPUT FEATURES/FEATURE ENGINEERING##

#Imports data
stocks = pd.read_excel("ASX200top10.xlsx", sheet_name = "Bloomberg raw", header = [0,1], index_col = 0)

clients = pd.read_excel("Client_Details.xlsx", sheet_name = "Data", index_col = 0)

indicators = pd.read_excel("Economic_Indicators.xlsx")

news = pd.read_json("news_dump.json")

#Removes all other columns except the PX_LAST column
stocks = stocks.iloc[:, stocks.columns.get_level_values(1)=='PX_LAST']

stocks.columns = stocks.columns.droplevel(1)

#Isolates the ASX Index price from the rest of the data
asx = stocks['AS51 Index']

stocks = stocks.drop('AS51 Index', axis = 1)

#Calculate returns and covariance matrix
rets = np.log(stocks / stocks.shift(1))

rets.head()

rets.mean() * 252

rets.cov() * 252

#Plots the correlation heatmap
fig, ax = plt.subplots(figsize=(18,10))
sns.heatmap(rets.cov() * 252, cmap = "coolwarm", annot = True)

##MODEL DESIGN##

#Sets the number of assets
noa = len(stocks.columns)

#Setup random porfolio weights, will sum up to 1
weights = np.random.random(noa)
weights /= np.sum(weights)

#Simulates 2500 combinations of portfolio returns and volatilities
prets = []
pvols = []
for p in range (2500):
    weights = np.random.random(noa)
    weights /= np.sum(weights)
    prets.append(np.sum(rets.mean() * weights) * 252)
    pvols.append(np.sqrt(np.dot(weights.T,
                                np.dot(rets.cov() * 252, weights))))
prets = np.array(prets)
pvols = np.array(pvols)

#Plots the 2500 combinations as a scatter plot, coloured by Sharpe ratio
plt.figure(figsize=(18, 10))
plt.scatter(pvols, prets, c=prets / pvols, marker='o')
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.show()

#calculates portfolio returns and volatility
def statistics(weights):
    weights = np.array(weights)
    pret = np.sum(rets.mean() * weights) * 252
    pvol = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))
    return np.array([pret, pvol, pret / pvol])

#Returns the Sharpe Ratio
def min_func_sharpe(weights):
    return -statistics(weights)[2]

#Minimises the negative Sharpe Ratio, which maximises the Sharpe Ratio
cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
bnds = tuple((0, 1) for x in range(noa))
noa * [1. / noa,]

opts = sco.minimize(min_func_sharpe, noa * [1. / noa,], method='SLSQP',
                    bounds=bnds, constraints=cons)

#Prints the portfolio with the maximum Sharpe ratio
print("***Maximization of Sharpe Ratio***")
#print(opts)
print(opts['x'].round(3))
print(statistics(opts['x']).round(3))

#select securities that will minimise the portfolio variance
def min_func_variance(weights):
    return statistics(weights)[1] ** 2

optv = sco.minimize(min_func_variance, noa * [1. / noa,], method='SLSQP',
                    bounds=bnds, constraints=cons)

#Prints the portfolio with the smallest variance
print("****Minimizing Variance***")
#print(optv)
print(optv['x'].round(3))
print(statistics(optv['x']).round(3))

#Returns the portfolio standard deviation
def min_func_port(weights):
    return statistics(weights)[1]

#Finds the portfolio that minimises standard deviation for each level of returns 
bnds = tuple((0, 1) for x in weights)
trets = np.linspace(0.0, 0.25, 50)
tvols = []
for tret in trets:
    cons = ({'type': 'eq', 'fun': lambda x:  statistics(x)[0] - tret},
            {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
    res = sco.minimize(min_func_port, noa * [1. / noa,], method='SLSQP',
                       bounds=bnds, constraints=cons)
    tvols.append(res['fun'])
tvols = np.array(tvols)

plt.figure(figsize=(18, 10))
plt.scatter(pvols, prets,
            c=prets / pvols, marker='o')
# random portfolio composition
plt.scatter(tvols, trets,
            c=trets / tvols, marker='x')
# efficient frontier
plt.plot(statistics(opts['x'])[1], statistics(opts['x'])[0],
         'r*', markersize=15.0)
# portfolio with highest Sharpe ratio
plt.plot(statistics(optv['x'])[1], statistics(optv['x'])[0],
         'y*', markersize=15.0)
# minimum variance portfolio
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.show()

#Finds points along the efficient frontier
ind = np.argmin(tvols)
evols = tvols[ind:]
erets = trets[ind:]
tck = sci.splrep(sorted(evols), erets)

def f(x):
    #Splines approximation of the efficient frontier function 
    return sci.splev(x, tck, der=0)
def df(x):
    #First derivative of efficient frontier function
    return sci.splev(x, tck, der=1)

def equations(p, rf=0.01):
    #these equations are used to find the parameters for the efficient fronter
    #and the capital market line
    eq1 = rf - p[0]
    eq2 = rf + p[1] * p[2] - f(p[2])
    eq3 = p[1] - df(p[2])
    return eq1, eq2, eq3

#returns an array of three floats; the first two represent the efficient frontier 
#and the third represents the capital market line.
opt = sco.fsolve(equations, [0.01, 0.5, 0.15])
print(opt)
print(np.round(equations(opt), 6))

#Plots the 2,500 random portfolios, the efficient frontier, and the capital market line 
plt.figure(figsize=(18, 10))
plt.scatter(pvols, prets,
            c=(prets - 0.01) / pvols, marker='o')
# random portfolio composition
plt.plot(evols, erets, 'g', lw=4.0)
# efficient frontier
cx = np.linspace(0.0, 0.3)
plt.plot(cx, opt[0] + opt[1] * cx, lw=1.5)
# capital market line
plt.plot(opt[2], f(opt[2]), 'r*', markersize=15.0)
plt.grid(True)
plt.axhline(0, color='k', ls='--', lw=2.0)
plt.axvline(0, color='k', ls='--', lw=2.0)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.show()

#Calculates the Optimal Tangent Portfolio
cons = ({'type': 'eq', 'fun': lambda x:  statistics(x)[0] - f(opt[2])},
        {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
res = sco.minimize(min_func_port, noa * [1. / noa,], method='SLSQP',
                   bounds=bnds, constraints=cons)
print("***Optimal Tangent Portfolio***")
print(res['x'].round(3))

#Mean and standard deviation of the optimal tangent portfolio
ret_rp = opt[2]
vol_rp = float(f(opt[2]))

#Calculates the risky share given a risk appetite
def y_star(ret_rp, vol_rp, A, rf = 0.01):
    return (ret_rp - rf) / (A * (vol_rp ** 2))

clients['risk_profile'] = 11 - clients['risk_profile']

#Calculates the non-adjusted portfolio weights.
for i in clients['risk_profile']:
    y = y_star(ret_rp, vol_rp, i)
    print("Invest %.3lf in the risk-free asset and the rest in the following portfolio:" % max(1 - y, 0))
    print((res['x'] * max(y,0)).round(3))
    print()

##SENTIMENT ANALYTICS##
import io
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

#Code that plots a graph of a metric against the number of epochs.
def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
    plt.show()
#Imports the IMDB movie reviews data for use 
dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                          as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

#The dataset info includes an encoder
encoder = info.features['text'].encoder

print('Vocabulary size: {}'.format(encoder.vocab_size))

#Sets the hyperparameters buffer size and batch sizer wwwwwwwwww
BUFFER_SIZE = 10000
BATCH_SIZE = 64

#Input features stage: Prepares the data for training
#by creating batches of these encoded strings
#These batches are zero-padded 
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE, ([-1],[]))
test_dataset = test_dataset.padded_batch(BATCH_SIZE, ([-1],[]))

#Constructs the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])

#Compiles the model
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

#fits the model
history = model.fit(train_dataset, epochs=1,
                    validation_data=test_dataset,
                    validation_steps=30)

#test_loss, test_acc = model.evaluate(test_dataset)

#print('Test Loss: {}'.format(test_loss))
#print('Test Accuracy: {}'.format(test_acc))

#Masks the padding applied to the sequences, which can lead to skew 
#if trained on padded sequences and tested on un-padded sequences
def pad_to_size(vec, size):
    zeros = [0] * (size - len(vec))
    vec.extend(zeros)
    return vec

def sample_predict(sample_pred_text, pad):
    encoded_sample_pred_text = encoder.encode(sample_pred_text)

    if pad:
        encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
    encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
    predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))

    return (predictions)

# predict on a sample text without padding.

sample_pred_text = ('The movie was not good. The animation and the graphics '
                    'were terrible. I would not recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=False)
print(predictions)

# predict on a sample text with padding

sample_pred_text = ('The movie was not good. The animation and the graphics '
                    'were terrible. I would not recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=True)
print(predictions)

prediction_dataset = news["Headline"]

prediction_values = []

#Runs the model on the news headlines
for quote in prediction_dataset:
    prediction_values.append(float(sample_predict(quote, pad=True)))

#Plots the raw sentiment values obtained as a histogram.    
plt.hist(prediction_values, bins = 20)
plt.xlabel("Raw Sentiment Value")
plt.ylabel("Count")
plt.show()

#scales the predictions to a number between 0 and 1
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale

prediction_values = np.array(prediction_values)

scaled_sentiments = minmax_scale(prediction_values)

#Plots the scaled sentiment values obtained as a histogram.   
plt.hist(scaled_sentiments, bins = 20)
plt.xlabel("Scaled Sentiment Value")
plt.ylabel("Count")
plt.show()

#Adds the scaled sentiments to the news DataFrame
news["Sentiments"] = scaled_sentiments

#Fixes the formatting of the Date/Time field
datetime = news["Date/Time"]

adjusted_datetime = []

for t in datetime:
    new = t.replace("'", "20")
    adjusted_datetime.append(new)

news['Adjusted DateTime'] = pd.Series(adjusted_datetime)

news['Adjusted DateTime'] = pd.to_datetime(news['Adjusted DateTime'])

news['Date'] = news['Adjusted DateTime'].dt.date

news = news.drop("Date/Time", axis = 1)

news = news.drop("Adjusted DateTime", axis = 1)

#Upsamples the consumer sentiment index
sent_index = pd.read_csv("consumer_sentiment_index.csv", parse_dates=[0], index_col = 0)

sent_index = sent_index.resample('D')

sent_index = sent_index.interpolate()

sent_index.head()

#Plots Consumer Sentiment Index over time
plt.figure(figsize=(12,8))
plt.plot(sent_index)
plt.xlabel("Date")
plt.ylabel("Consumer Sentiment Index")
plt.show() 
#Consumer Sentiment Index fluctuates quite dramatically, reaching a low of approximately 75 in May 2020

#Plots the sentiment values over time for each stock
for ticker in news['Equity'].unique():
    plt.plot(news[news["Equity"] == ticker]["Date"], news[news["Equity"] == ticker]["Sentiments"])
    plt.xlabel("Date")
    plt.ylabel(ticker)
    plt.show()

#Calculates mean consumer sentiment
mean_consumer_sentiment = np.mean(sent_index).item()
print(mean_consumer_sentiment)

#Adjusts the sentiment values to account for the Consumer Sentiment Index
adj_sentiments = []

for i in news.index:
    sentiment = news['Sentiments'][i]
    date = news['Date'][i]
    csi = sent_index['Consumer Sentiment Index'][date]
    adj_sentiment = sentiment * mean_consumer_sentiment / csi
    adj_sentiments.append(adj_sentiment)

news['Adjusted Sentiments'] = pd.Series(adj_sentiments)

news = news.drop(['Sentiments'], axis = 1)

#Calculates the mean sentiment for each stock
mean_sentiments = news.groupby('Equity').mean()

#Calcualtes the sentiment multiplier
mean_sentiments['Sentiment Multiplier'] = mean_sentiments['Adjusted Sentiments'] / 0.5

##MODEL IMPLEMENTATION##
#Calculates and plots the adjusted portfolio weights
for risk in clients['risk_profile']:
    y = y_star(ret_rp, vol_rp, risk)
    pre_adjust = res['x'] * max(y, 0)
    post_adjust = pre_adjust * mean_sentiments['Sentiment Multiplier']
    print("Invest %.3lf in the risk-free asset and the rest in the following portfolio:" % max(1 - sum(post_adjust), 0))
    rounded = post_adjust.round(3)
    print(rounded)
    print()
    
    reduced = rounded[rounded != 0]
    labels = list(reduced.index)
    labels.append("Risk-free asset")
    sizes = list(reduced)
    sizes.append(max(1 - sum(post_adjust), 0))
    
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    ax1.set_title("Complete Portfolio, Risk Aversion Factor %.2lf" % risk, pad=20)
    plt.show()

    
rounded = rounded / max(sum(post_adjust), 0)

reduced = rounded[rounded != 0]
labels = list(reduced.index)
sizes = list(reduced)

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')
plt.title("Optimal Tangent Portfolio")
plt.show()

def absolute_value(val):
    a  = np.round(val/100.*np.sum(sizes), 0)
    return a

#Minimum viable product. Allows users to set their risk appetite and capital.
print("Welcome to Pautolio!")

#Set risk appetite
try:
    risk = float(input("How risk averse are you from 1 to 10? (Higher is more averse) "))
    assert risk >= 1 and risk <= 10
except:
    raise ValueError("Input must be a number from 1 to 10")

#Calculates adjusted portfolio weights    
y = y_star(ret_rp, vol_rp, risk)
pre_adjust = res['x'] * max(y, 0)
post_adjust = pre_adjust * mean_sentiments['Sentiment Multiplier']
rounded = post_adjust.round(3)

#Converts the portfolio weights to real dollar values based on the user input
try:
    valid = True
    capital = float(input("How much would you like to invest? "))
    assert capital > 0
except:
    valid = False
    print("Invalid input detected. Returning percentage values.")
    capital = 1

#Plots the portfolio    
reduced = rounded[rounded != 0] * capital
labels = list(reduced.index)
labels.append("Risk-free asset")
sizes = list(reduced)
sizes.append(capital * max(1 - sum(post_adjust), 0))

fig1, ax1 = plt.subplots()

if valid:
    ax1.pie(sizes, labels=labels, autopct=absolute_value, startangle=90)
else:
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    
ax1.axis('equal')
ax1.set_title("Complete Portfolio, Risk Aversion Factor %.3lf" % risk, pad=20)
plt.show()

#Plots the stock prices over time
for ticker in stocks:
    plt.plot(stocks.index, stocks[ticker])
    plt.xlabel("Date")
    plt.ylabel(ticker)
    plt.show()

#Saves the RNN model    
model.save('model_with_dropout')

