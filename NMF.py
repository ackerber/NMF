#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.decomposition as skd
import sklearn.metrics as skm
import shap
from sklearn.linear_model import LinearRegression


# Load data

name = ["user","movie","rating","timestamp"]
movies = pd.read_csv("movies.tsv", sep = "\t", header = None)
train = pd.read_csv("train.tsv", sep = "\t", names = name)
test = pd.read_csv("test.tsv", sep = "\t", names = name)
valid = pd.read_csv("valid.tsv", sep = "\t", names = name)
users = pd.read_csv("users.tsv", sep = "\t", header = None)


# Explicit -> Implicit

train.rating = (train.rating >= 4).astype(int)
test.rating = (test.rating >= 4).astype(int)
valid.rating = (valid.rating >= 4).astype(int)

# Modeling

train = train.drop(columns = ['timestamp'])

um = train.pivot_table(index='user', columns='movie', values='rating').rename_axis(index=None, columns=None)

model = skd.NMF(n_components = 10, init='random', random_state=0, tol = 0.05, max_iter = 100000)
W = model.fit_transform(um.fillna(0))
H = model.components_
WH = np.matmul(W,H)


# Shapley Value
surrogate_model = LinearRegression()
surrogate_model.fit(WH, um.fillna(0))
explainer = shap.Explainer(surrogate_model, WH)
shap_values = explainer.shap_values(WH[0])

shap.summary_plot(shap_values, WH[0], plot_type='bar', max_display=11)

# finds the movie number
p = pd.DataFrame(WH[0])
r = [0.0815859,0.024876,0.111997,0.125398,0.11517,0.074799, 0.22046, 0.14135, 0.084296, 0.15449]
p = p[abs(p-r[0]) < 0.00001]
p.dropna()

# finds the movie names
[movies[1].iloc[983], movies[1].iloc[2701], movies[1].iloc[33], movies[1].iloc[1452], movies[1].iloc[2547],
movies[1].iloc[1121], movies[1].iloc[1751], movies[1].iloc[2553], movies[1].iloc[943], movies[1].iloc[3126]]

# finding comedy genre percentage
m = movies[2]
m = pd.Series(list(m))
sum(m.str.count("Comedy")) / len(m)

# finding movie genres
[movies[2].iloc[983], movies[2].iloc[2701], movies[2].iloc[33], movies[2].iloc[1452], movies[2].iloc[2547],
movies[2].iloc[1121], movies[2].iloc[1751], movies[2].iloc[2553], movies[2].iloc[943], movies[2].iloc[3126]]


# Get each user's most recent rating

rates = []
stamps = test.pivot_table(index='user', columns='movie', 
                          values='timestamp').rename_axis(index=None, columns=None)
rate = test.pivot_table(index='user', columns='movie', 
                        values='rating').rename_axis(index=None, columns=None)

for i in range(0,6040):
    s = stamps.iloc[i].idxmin()
    rates.append(rate[s][i])


# Finding the Hit Rate
# For each user, randomly sample 100 movies which they did not rate from the training set. Find the new estimated ratings of those movies. Add the  user's most recent rating to the list. From there, rank those movies and take the top or bottom 10 of them - top if the most recent rating was a 1, bottom if the most recent rating was a 0. Find whether the most recent rating for each is in the top/bottom ranking. Find the percentage of times this occurs, and you have the hit rate.

hr = 0
ndcg = 0
for i in range(0,6040):
    
    li = []
    curr = um.iloc[i]
    for j in range(0,3648):
        if np.isnan(curr[j]):
            li.append(j)

    dcurr = pd.DataFrame(WH).iloc[i]
    dcurr = dcurr[li]
    rank = pd.DataFrame(dcurr.sample(101)).reset_index()
    rank[100:] = rates[i]

    if rates[i] == 1:
        rank = rank.sort_values(by=[i], ascending = False)[:10]
        s = pd.Series(list(rank['index']))
        if 1 in s:
            hr = hr+1

        if rank.iloc[0][i] == 1:
            true_rel = rank[i]
        else: 
            true_rel = pd.concat([pd.DataFrame([[0,1]], columns = ['index',i]), rank])
            true_rel = true_rel[:10][i]
        ndcg = ndcg + skm.ndcg_score(np.asarray([true_rel]), np.asarray([rank[i]]))

    else:
        rank = rank.sort_values(by=[i], ascending = True)[:10]
        s = pd.Series(list(rank['index']))
        if 0 in s:
            hr = hr+1

        if rank.iloc[9][i] == 0:
            true_rel = rank[i]
        else: 
            true_rel = pd.concat([rank, pd.DataFrame([[0,0]], columns = ['index',i])])
            true_rel = true_rel[1:][i]
        ndcg = ndcg + skm.ndcg_score(np.asarray([true_rel]), np.asarray([rank[i]]))


# Find average hit rate and NDCG

hr/6040

ndcg/6040






