# This class will extract the information from the .csv database

# IMPORTS
import tkFileDialog

from tkinter import *
import tkinter as tk
from PIL import ImageTk, ImageFilter  # module to add a background picture
from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import numpy as np
import pandas as pd  # processing the .csv
import matplotlib
from collections import Counter
import seaborn as sns  # statistical data visualization
from sklearn import preprocessing, svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
import statsmodels.api as sm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from tkinter import messagebox, filedialog


matplotlib.use('TkAgg')
import matplotlib.pyplot as plot

FILE_NAME = '/Users/pabloaramburu/PycharmProjects/IMDB-Pablo/IMDB_database.csv'
fBullet = u"\u2022" + " "
sBullet = "\n" + u"\u2022" + " "
data = pd.DataFrame.empty
dataKNN = pd.DataFrame.empty

# See the whole output
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 45)
pd.set_option('display.width', 1000)


def readDatabaseRaw(filename):
    print("\n====================== Importing data  ======================")
    dataframe = pd.read_csv(filename)
    print("\nData has been imported")
    # print(data.head())
    return dataframe


def getDataInfo(filename):
    print("\n====================== Getting info from dataFrame ======================")
    print("\nFirst rows")
    print(filename.head())
    print("\nSize of data (rows, columns): " + str(filename.shape))
    columns = " "
    c = 0
    for s in filename.columns:
        if c < len(filename.columns) - 1:
            columns = columns + s + " , "
        else:
            columns = columns + s
        c = c + 1
    print("\nColumns:" + columns)
    # print (filename.info())


def removeNullValues(filename):
    a = filename.shape
    print("\n====================== Removing null values from dataFrame ======================")
    print("\nCurrent size of the dataframe is: " + str(filename.shape))
    print("\nPrinting how many null values are there on each category")
    print(filename.isnull().sum().sort_values(ascending=False)[:10])
    print("...")
    # how = 'any' deletes the row if any NA value appears
    filename.dropna(how='any', inplace=True);
    print("\nDone! " + str(a[0] - filename.shape[0]) + " rows has been deleted since they contained null values")
    print("\nNew size of the dataframe is: " + str(filename.shape))
    return filename


# ======================================== DATA EXPLORATION ========================================

def countCountryTimes(data):
    country = []
    # Extraction of each country and add it to the array
    for i in data['country']:
        country.extend(i.split(","))
    ntimescountry = Counter(country)
    return ntimescountry


def countGenreTimes(data):
    country = []
    for i in data['genres']:
        country.extend(i.split("|"))
    ntimesgenre = Counter(country)
    return ntimesgenre


# ======================================== COUNTRIES ========================================


def moviesCountry(ntimescountry):
    ntimescountry = ntimescountry.most_common()

    # Creation of dataframe with pandas
    countriesDataFrame = pd.DataFrame(ntimescountry, columns=["country", "ntimes"])

    # Plotting the dataframe. I reset the index of those countries with ntimes>2 in order to
    # simplify the graphic.
    lastCountriesDF = countriesDataFrame[countriesDataFrame["ntimes"] > 2].reset_index(drop=True)
    #print(lastCountriesDF)
    # print(lastCountriesDF)
    plot.figure(1)
    # plot.subplots(figsize=(13, 7))
    g = sns.barplot(x="country", y="ntimes", data=lastCountriesDF)
    plot.xticks([i for i in range(len(lastCountriesDF))], lastCountriesDF["country"], rotation=40)
    plot.title("Number of movies per country")
    plot.xlabel("Country")
    plot.ylabel("Number of movies")
    plot.tight_layout()
    # plot.show()


def avgCountry(data, ntimescountry):
    ntimescountry = ntimescountry.most_common()  # sort countries
    topCountries = []
    limit = 0
    for i, c in enumerate(ntimescountry):
        if (limit >= 26):
            break
        topCountries.append(c[0])
        limit = limit + 1
    for i in topCountries:
        data[i] = data["country"].map(lambda x: 1 if i in str(x) else 0)

    avgCountry = []
    for i in topCountries:
        avgCountry.append([i, data["imdb_score"][data[i] == 1].mean()])
    avgCountry = pd.DataFrame(avgCountry, columns=["country", "mean"])

    scaler2 = preprocessing.MinMaxScaler(feature_range=(0, 10))
    avgCountry["scaled"] = scaler2.fit_transform(avgCountry["mean"].values.reshape(-1, 1))

    #print(avgCountry)
    # Plotting Mean Rating per Country
    plot.figure(2)
    # plot.subplots(figsize=(13, 7))
    g = sns.barplot(x="country", y="mean", data=avgCountry)
    plot.title("Mean rating per country")
    plot.ylabel("Mean rating")
    plot.xlabel("Country")
    plot.xticks(rotation=45)
    plot.tight_layout()
    # plot.show()

    # Plotting Mean Scaled Rating for each country in relation with the others
    plot.figure(3)
    # plot.subplots(figsize=(13, 7))
    sns.barplot(x="country", y="scaled", data=avgCountry)
    plot.title("Scaled mean rating per country")
    plot.ylabel("Scaled mean rating")
    plot.xlabel("Country")
    plot.xticks(rotation=45)
    plot.tight_layout()
    # plot.show()


def medianCountry(data, ntimescountry):
    # This method computes the median of each contry and then assigns a 0 to the lowest and 10 to the highest.
    # The rest are distributed along 0-10.
    ntimescountry = ntimescountry.most_common()  # sort countries
    topCountries = []
    limit = 0
    # Selection of the 15 most common countries to simplify the graphs
    for i, c in enumerate(ntimescountry):
        if (limit >= 26):
            break
        topCountries.append(c[0])
        limit = limit + 1

    # change the value of country to 1 if exists in order to be able later to select them from data
    for i in topCountries:
        data[i] = data["country"].map(lambda x: 1 if i in str(x) else 0)

    ratingCountry = []
    for i in topCountries:
        ratingCountry.append([i, data["imdb_score"][data[i] == 1].median()])

    ratingCountry = pd.DataFrame(ratingCountry, columns=["country", "median"])
    scaler2 = preprocessing.MinMaxScaler(feature_range=(0, 10))
    ratingCountry["scaled"] = scaler2.fit_transform(ratingCountry["median"].values.reshape(-1, 1))

    #print(ratingCountry)
    # Plotting Median Rating per Country
    plot.figure(4)
    sns.barplot(x="country", y="median", data=ratingCountry)

    plot.title("Median rating per country")
    plot.ylabel("Median rating")
    plot.xlabel("Country")
    plot.xticks(rotation=45)
    plot.tight_layout()
    # plot.show()

    # Plotting Median Scaled Rating for each country in relation with the others
    plot.figure(5)
    sns.barplot(x="country", y="scaled", data=ratingCountry)
    plot.title("Scaled median rating per country")
    plot.ylabel("Scaled median rating")
    plot.xlabel("Country")
    plot.xticks(rotation=45)
    plot.tight_layout()
    # plot.show()


# ======================================== GENRES ========================================

def moviesGenre(ntimesgenre):
    ntimesgenre = ntimesgenre.most_common()
    # Creation of dataframe with pandas
    genresDataFrame = pd.DataFrame(ntimesgenre, columns=["genre", "ntimes"])
    # Plotting the dataframe. I reset the index of those genres with ntimes>2 in order to
    # simplify the graphic.
    lastCountriesDF = genresDataFrame[genresDataFrame["ntimes"] > 151].reset_index(drop=True)
    plot.figure(6)
    # plot.subplots(figsize=(13, 7))
    g = sns.barplot(x="genre", y="ntimes", data=lastCountriesDF)
    plot.xticks([i for i in range(len(lastCountriesDF))], lastCountriesDF["genre"], rotation=45)
    plot.title("Number of movies per genre")
    plot.xlabel("Genre")
    plot.ylabel("Number of movies")
    plot.tight_layout()
    # plot.show()


def avgGenre(data, ntimesgenre):
    ntimesgenre = ntimesgenre.most_common()  # sort genres
    topGenres = []
    limit = 0
    for i, c in enumerate(ntimesgenre):
        if (limit >= 15):
            break
        topGenres.append(c[0])
        limit = limit + 1

    for i in topGenres:
        data[i] = data["genres"].map(lambda x: 1 if i in str(x) else 0)

    avgGenre = []

    for i in topGenres:
        avgGenre.append([i, data["imdb_score"][data[i] == 1].mean()])
    avgGenre = pd.DataFrame(avgGenre, columns=["genres", "mean"])

    scaler2 = preprocessing.MinMaxScaler(feature_range=(0, 10))
    avgGenre["scaled"] = scaler2.fit_transform(avgGenre["mean"].values.reshape(-1, 1))
    print(avgGenre)
    # Plotting Mean Rating per Genre
    plot.figure(7)
    sns.barplot(x="genres", y="mean", data=avgGenre)
    plot.title("Mean rating per genre")
    plot.ylabel("Mean rating")
    plot.xlabel("Genre")
    plot.xticks(rotation=45)
    plot.tight_layout()
    # plot.show()

    # Plotting Mean Scaled Rating for each genre in relation with the others
    plot.figure(8)
    sns.barplot(x="genres", y="scaled", data=avgGenre)
    plot.title("Scaled mean rating per genre")
    plot.ylabel("Scaled mean rating")
    plot.xlabel("Genre")
    plot.xticks(rotation=45)
    plot.tight_layout()
    # plot.show()


def medianGenre(data, ntimesgenre):
    ntimesgenre = ntimesgenre.most_common()  # sort countries
    topGenres = []
    limit = 0
    for i, c in enumerate(ntimesgenre):
        if (limit >= 15):
            break
        topGenres.append(c[0])
        limit = limit + 1

    # change the value of country to 1 if exists in order to be able later to select them from data
    for i in topGenres:
        data[i] = data["genres"].map(lambda x: 1 if i in str(x) else 0)

        ratingGenre = []
    for i in topGenres:
        ratingGenre.append([i, data["imdb_score"][data[i] == 1].median()])

    ratingGenre = pd.DataFrame(ratingGenre, columns=["genres", "median"])
    scaler2 = preprocessing.MinMaxScaler(feature_range=(0, 10))
    ratingGenre["scaled"] = scaler2.fit_transform(ratingGenre["median"].values.reshape(-1, 1))
    print(ratingGenre)
    # Plotting Median Rating per Genre
    plot.figure(9)
    sns.barplot(x="genres", y="median", data=ratingGenre)
    plot.title("Median rating per genre")
    plot.ylabel("Median rating")
    plot.xlabel("Genre")
    plot.xticks(rotation=45)
    plot.tight_layout()
    # plot.show()

    # Plotting Median Scaled Rating for each genre in relation with the others
    plot.figure(10)
    sns.barplot(x="genres", y="scaled", data=ratingGenre)
    plot.title("Scaled median rating per genre")
    plot.ylabel("Scaled median rating")
    plot.xlabel("Genre")
    plot.xticks(rotation=45)
    plot.tight_layout()
    # plot.show()


def dataExploration(data):
    print("\n====================== Exploration of Data ======================")
    ntimescountry = countCountryTimes(data)
    ntimesgenre = countGenreTimes(data)

    ''' Analysis per Country '''
    moviesCountry(ntimescountry)
    avgCountry(data, ntimescountry)
    medianCountry(data, ntimescountry)

    ''' Analysis per Genre '''
    moviesGenre(ntimesgenre)
    avgGenre(data, ntimesgenre)
    medianGenre(data, ntimesgenre)
    plot.show()


# ======================================== CORRELATION ========================================

def correlation(data):
    # ONLY DONE WITH THOSE COLUMNS THAT CONTAIN NUMBERS, NOT STRINGS
    corrmat = data.corr()
    top_corr_features = corrmat.index[corrmat["imdb_score"] > 0.1]
    sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="RdYlGn", linewidths=.5)
    #sns.heatmap(corrmat, annot=True, cmap="RdYlGn") RdYlGn
    #plot.tight_layout()
    plot.xticks(rotation=45)
    plot.title("Correlation between variables")
    plot.show()
    corrmat.sort_values(["imdb_score"], ascending=False, inplace=True)
    print("\n====================== Correlation between IMDB score and variables ======================")
    print("\n")
    print(corrmat.imdb_score)


# ======================================== PREDICTION ALGORITHMS ========================================

def linearRegressionModel(data, accuracies, isSummary):
    print("\n====================== Using Linear Regression Model ======================")
    # Drop every cell that is a string
    x = data.drop(['color', 'director_name', 'actor_2_name', 'genres', 'actor_1_name', 'movie_title', 'actor_3_name',
                   'plot_keywords', 'movie_imdb_link', 'language', 'country', 'content_rating',
                   'aspect_ratio', 'imdb_score'], axis=1)
    y = data.imdb_score

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.20)
    print("\n Data has been divided into train and test sets")
    # print (X_train.shape)
    # print (y_train.head())
    # print (X_test)
    # print (y_test.head())

    model = sm.OLS(y_train.astype(float), X_train.astype(float))  # Least Squares Model
    results = model.fit()  # Trains the model
    # print (results.summary())

    print("\n Computing predictions with test set")
    pred = results.predict(X_test)  # Compute predictions to x_test using the results of training
    h = pd.DataFrame(pred)
    h = h.round(1)
    g = pd.DataFrame(y_test)
    # g = g.reset_index(drop=True)
    h['rating'] = g.astype(float)
    h.columns = ['predictedScore', 'actualScore']
    h['difference'] = abs(h.predictedScore - h.actualScore)
    print("\n Predictions computed. An example is given below.\n")
    print(h[:2])

    preds = (h.difference < 1.0).sum()
    total = (h.difference).count()
    print("\n ----> Results")
    print("Predictions within 1 point: %d" % (preds))
    print('Total Predictions: %d' % (total))
    print ('Accuracy: %f' % (float(preds) / float(total)))
    accuracies["Linear Regression"] = round(float(preds) / float(total), 2)
    errors = abs(pred - y_test)

    print('Mean Absolute Error: ' + str(round(np.mean(errors), 2)))

    if isSummary == False:
        plot.hist(h.difference)
        plot.xlim(0)
        plot.xlabel("Difference")
        plot.ylabel("Number of movies")
        plot.suptitle("Linear Regression")
        plot.title("Difference between predicted and actual values")
        # Comment this line if not using tkinter
        text = fBullet + "Predictions within +/- 1 point: " + str(preds) + sBullet + "Total Predictions: " \
               + str(total) + sBullet + "Accuracy: " + str(float(preds) / float(total)) + \
               sBullet + "Mean Absolute Error: " + str(round(np.mean(errors), 2))
        messagebox.showinfo("Logisitic Regression Results", text)
        plot.show()


def kNearestNeighbors(data, accuracies, isSummary):
    print("\n====================== Using K-Nearest Neighbors Model ======================")

    h = data
    h_return = data
    bins = [0.0, 3.0, 5.0, 7.0, 8.0, 10.0]
    groups = ['E', 'D', 'C', 'B', 'A']
    h['categories'] = pd.cut(h.imdb_score, bins, labels=groups)
    # print(h)

    x = h.drop(['color', 'director_name', 'actor_2_name', 'genres', 'actor_1_name', 'movie_title', 'actor_3_name',
                'plot_keywords', 'movie_imdb_link', 'language', 'country', 'content_rating',
                'aspect_ratio', 'imdb_score', 'categories'], axis=1)
    y = h.categories

    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.20)
    k_range = range(25, 40)
    k_scores = []

    print("\n Computing predictions with test set")
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        pred = knn.predict(X_test)
        z = metrics.accuracy_score(y_test, pred)
        k_scores.append(z)
        if(z>0.621010):
            print(k)

    k_scores = pd.DataFrame(k_scores)
    print('Maximum Accuracy for KNN: %f' % k_scores.max())
    accuracies["K-Nearest Neighbors"] = round(float(k_scores.max()), 2)

    if isSummary == False:
        plot.plot(k_range, k_scores)
        plot.xlabel('K value')
        plot.ylabel('Accuracy')
        plot.title("K-Nearest Neighbors")
        text = fBullet + 'Accuracy: ' + str(k_scores[0].max())
        # Comment this line if not using tkinter
        messagebox.showinfo("K-Nearest Neighbors Results", text)
        plot.show()

    return h_return


def naiveBayesGaussian(data, accuracies, isSummary):
    print("\n====================== Using Naive Bayes Gaussian Model ======================")
    x = data.drop(['color', 'director_name', 'duration', 'actor_3_facebook_likes', 'actor_2_name', 'gross', 'genres',
                   'facenumber_in_poster', 'actor_1_name', 'movie_title', 'actor_3_name',
                   'plot_keywords', 'movie_imdb_link', 'language', 'country', 'content_rating',
                   'aspect_ratio', 'imdb_score'], axis=1)
    y = data.imdb_score
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.20)
    print("\n Data has been divided into training and testing sets")

    model_gauss = GaussianNB()
    results_g = model_gauss.fit(X_train, (10 * y_train).astype(int))  # multiply by 10 to keep decimals and to int
    print("\n Computing predictions with test set")
    pred_gauss = results_g.predict(X_test)

    h_gauss = pd.DataFrame(pred_gauss)  # PREDICTIONS
    h_gauss = h_gauss.round(1) / 10.0
    g = pd.DataFrame(y_test)  # REAL SCORES
    g = g.reset_index(drop=True)

    h_gauss['rating'] = g
    h_gauss.columns = ['predictedScore', 'actualScore']
    h_gauss['difference'] = abs(h_gauss.predictedScore - h_gauss.actualScore)

    print("\n Predictions computed. An example is given below.\n")
    print(h_gauss[:2])

    preds = (h_gauss.difference < 1.0).sum()
    total = (h_gauss.difference).count()
    print("\n ----> Results")
    print("Predictions within +/- 1 point of difference: %d" % (preds))
    print('Total Predictions: %d' % (total))
    print ('Accuracy for Naive Bayes Gaussian Model: %f' % (float(preds) / float(total)))

    accuracies["Naive Bayes Gaussian"] = round(float(preds) / float(total), 2)
    errors = abs(pred_gauss / 10.0 - y_test)
    print('Mean Absolute Error: ' + str(round(np.mean(errors), 2)))
    if isSummary == False:
        plot.hist(h_gauss.difference)
        plot.xlim(0)
        plot.xlabel("Difference")
        plot.ylabel("Quantity")
        plot.suptitle("Naive Bayes Gaussian Model")
        plot.title("Difference between predicted and actual values")
        text = fBullet + "Predictions within +/- 1 point: " + str(preds) + sBullet + "Total Predictions: " \
               + str(total) + sBullet + "Accuracy: " + str(
            float(preds) / float(total)) + sBullet + "Mean Absolute Error: " \
               + str(round(np.mean(errors), 2))
        # Comment this line if not using tkinter
        messagebox.showinfo("Naive Bayes Gaussian Results", text)

        plot.show()


def naiveBayesBernoulli(data, accuracies, isSummary):
    print("\n====================== Using Naive Bayes Bernoulli Model ======================")

    x = data.drop(['color', 'director_name', 'duration', 'actor_3_facebook_likes', 'actor_2_name', 'gross', 'genres',
                   'facenumber_in_poster', 'actor_1_name', 'movie_title', 'actor_3_name',
                   'plot_keywords', 'movie_imdb_link', 'language', 'country', 'content_rating',
                   'aspect_ratio', 'imdb_score'], axis=1)
    y = data.imdb_score
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.20)
    print("\n Data has been divided into training and testing sets")

    model_bern = BernoulliNB()
    results_b = model_bern.fit(X_train, (10 * y_train).astype(int))
    print("\n Computing predictions with test set")
    pred_bern = results_b.predict(X_test)
    h_bern = pd.DataFrame(pred_bern)
    h_bern = h_bern.round(1) / 10.0
    g = pd.DataFrame(y_test)  # REAL SCORES
    g = g.reset_index(drop=True)
    h_bern['rating'] = g
    h_bern.columns = ['predictedScore', 'actualScore']
    h_bern['difference'] = abs(h_bern.predictedScore - h_bern.actualScore)

    print("\n Predictions computed. An example is given below.\n")
    print(h_bern[:2])

    preds = (h_bern.difference < 1.0).sum()
    total = (h_bern.difference).count()
    print("\n ----> Results")
    print("Predictions within +/- 1 point of difference: %d" % (preds))
    print('Total Predictions: %d' % (total))
    print ('Accuracy for Naive Bayes Bernoulli Model: %f' % (float(preds) / float(total)))
    accuracies["Naive Bayes Bernoulli"] = round(float(preds) / float(total), 2)
    errors = abs(pred_bern / 10.0 - y_test)
    print('Mean Absolute Error: ' + str(round(np.mean(errors), 2)))
    if isSummary == False:
        plot.hist(h_bern.difference)
        plot.xlim(0)
        plot.xlabel("Difference")
        plot.ylabel("Quantity")
        plot.suptitle("Naive Bayes Bernoulli Model")
        plot.title("Difference between predicted and actual values")
        text = fBullet + "Predictions within +/- 1 point: " + str(preds) + sBullet + "Total Predictions: " + str(
            total) + sBullet + "Accuracy: " + str(float(preds) / float(total)) + sBullet + "Mean Absolute Error: " + \
               str(round(np.mean(errors), 2))
        # Comment this line if not using tkinter
        messagebox.showinfo("Naive Bayes Bernoulli Results", text)
        plot.show()


def supportVectorMachines(data, accuracies, isSummary):
    print("\n====================== Using Support Vector Machines Model ======================")

    x = data.drop(['color', 'director_name', 'duration', 'actor_3_facebook_likes', 'actor_2_name', 'gross', 'genres',
                   'facenumber_in_poster', 'actor_1_name', 'movie_title', 'actor_3_name',
                   'plot_keywords', 'movie_imdb_link', 'language', 'country', 'content_rating',
                   'aspect_ratio', 'imdb_score'], axis=1)
    y = data.imdb_score
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.20)
    print("\n Data has been divided into training and testing sets")

    clf = svm.SVC();
    clf.set_params(C=1000000000000)

    clf.fit(X_train, (10 * y_train).astype(int))
    print("\n Computing predictions with test set")
    prediction = clf.predict(X_test)

    h = pd.DataFrame(prediction)
    h = h.round(1) / 10.0

    g = pd.DataFrame(y_test)  # REAL SCORES
    g = g.reset_index(drop=True)
    h['rating'] = g
    h.columns = ['predictedScore', 'actualScore']
    h['difference'] = abs(h.predictedScore - h.actualScore)
    print("\n Predictions computed. An example is given below.\n")
    print(h[:2])
    preds = (h.difference < 1.0).sum()
    total = (h.difference).count()
    print("\n ----> Results")
    print("Predictions with +/- 1 point of difference: %d" % (preds))
    print('Total Predictions: %d' % (total))
    print ('Accuracy for SVM Model: %f' % (float(preds) / float(total)))

    accuracies["SVM"] = round(float(preds) / float(total), 2)
    errors = abs(prediction / 10.0 - y_test)
    print('Mean Absolute Error: ' + str(round(np.mean(errors), 2)))

    if isSummary == False:
        plot.hist(h.difference)
        plot.xlim(0)
        plot.xlabel("Difference")
        plot.ylabel("Quantity")
        plot.suptitle("Support Vector Machines")
        plot.title("Difference between predicted and actual values")
        text = fBullet + "Predictions within +/- 1 point: " + str(preds) + sBullet + "Total Predictions: " + str(
            total) + sBullet + "Accuracy: " + str(float(preds) / float(total)) + sBullet + "Mean Absolute Error: " + \
               str(round(np.mean(errors), 2))
        # Comment this line if not using tkinter
        messagebox.showinfo("Support Vector Machines Results", text)
        plot.show()


def randomForest(data, accuracies, isSummary):
    print("\n====================== Using Random Forests Regression ======================")

    x = data.drop(['color', 'director_name', 'duration', 'actor_3_facebook_likes', 'actor_2_name', 'gross', 'genres',
                   'facenumber_in_poster', 'actor_1_name', 'movie_title', 'actor_3_name',
                   'plot_keywords', 'movie_imdb_link', 'language', 'country', 'content_rating',
                   'aspect_ratio', 'imdb_score'], axis=1)
    y = data.imdb_score
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.20)
    print("\n Data has been divided into training and testing sets")
    dt = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0)
    dt.fit(X_train, y_train)
    print("\n Computing predictions with test set")
    prediction = dt.predict(X_test)
    h = pd.DataFrame(prediction)
    h = h.round(1)
    g = pd.DataFrame(y_test)  # REAL SCORES
    g = g.reset_index(drop=True)
    h['rating'] = g
    h.columns = ['predictedScore', 'actualScore']
    h['difference'] = abs(h.predictedScore - h.actualScore)
    print("\n Predictions computed. An example is given below.\n")
    print(h[:2])
    preds = (h.difference < 1.0).sum()
    total = (h.difference).count()
    print("\n ----> Results")
    print("Predictions with +/- 1 point of difference: %d" % (preds))
    print('Total Predictions: %d' % (total))
    print ('Accuracy for Random Forest Model: %f' % (float(preds) / float(total)))
    accuracies["Random Forest"] = round(float(preds) / float(total), 2)
    errors = abs(prediction - y_test)
    print('Mean Absolute Error: ' + str(round(np.mean(errors), 2)))

    if isSummary == False:
        plot.hist(h.difference)
        plot.xlim(0)
        plot.xlabel("Difference")
        plot.ylabel("Quantity")
        plot.suptitle("Random Forest Regression")
        plot.title("Difference between predicted and actual values")
        text = fBullet + "Predictions within +/- 1 point: " + str(preds) + sBullet + "Total Predictions: " + str(
            total) + sBullet + "Accuracy: " + str(float(preds) / float(total)) + sBullet + "Mean Absolute Error: " + \
               str(round(np.mean(errors), 2))
        # Comment this line if not using tkinter
        messagebox.showinfo("Random Forest Results", text)

        plot.show()


def main():
    dataframe = readDatabaseRaw(FILE_NAME)
    getDataInfo(dataframe)
    dataframeNoNull = removeNullValues(dataframe)
    # print(dataframeNoNull)
    dataExploration(dataframeNoNull)
    # correlation(dataframeNoNull)

    accuracies = {}

    ''' Studying different prediction algorithms'''
    # linearRegressionModel(dataframeNoNull, accuracies)
    # naiveBayesGaussian(dataframeNoNull, accuracies)
    # naiveBayesBernoulli(dataframeNoNull, accuracies)
    # supportVectorMachines(dataframeNoNull, accuracies)
    # randomForest(dataframeNoNull, accuracies)
   # kNearestNeighbors(dataframeNoNull, accuracies)

    # https://github.com/ankitesh97/IMDB-MovieRating-Prediction/blob/master/ml_model.py
    print("\n====================== Summary of Models ======================")
    print(accuracies)


def initialState(filename):
    dataframe = readDatabaseRaw(filename)
    getDataInfo(dataframe)
    dataframeNoNull = removeNullValues(dataframe)
    return dataframeNoNull


def getDataKNN(filename):
    dataframe = readDatabaseRaw(filename)
    getDataInfo(dataframe)
    dataframeNoNull = removeNullValues(dataframe)
    return dataframeNoNull


def analysisPerCountry(data):
    ntimescountry = countCountryTimes(data)
    moviesCountry(ntimescountry)
    avgCountry(data, ntimescountry)
    medianCountry(data, ntimescountry)
    plot.show()


def analysisPerGenre(data):
    ntimesgenre = countGenreTimes(data)
    moviesGenre(ntimesgenre)
    avgGenre(data, ntimesgenre)
    medianGenre(data, ntimesgenre)
    plot.show()


def summary(data, dataKNN, accuracies, isSummary):
    isSummary = True
    linearRegressionModel(data, accuracies, isSummary)
    naiveBayesGaussian(data, accuracies, isSummary)
    naiveBayesBernoulli(data, accuracies, isSummary)
    supportVectorMachines(data, accuracies, isSummary)
    randomForest(data, accuracies, isSummary)
    kNearestNeighbors(dataKNN, accuracies, isSummary)
    isSummary = False
    text = fBullet + "Linear Regression --> " + str(accuracies["Linear Regression"]) + sBullet \
           + "Naives Bayes Gaussian --> " + str(accuracies["Naive Bayes Gaussian"]) + sBullet \
           + "Naive Bayes Bernoulli --> " + str(accuracies["Naive Bayes Bernoulli"]) + sBullet \
           + "Support Vector Machines --> " + str(accuracies["SVM"]) + sBullet \
           + "Random Forest --> " + str(accuracies["Random Forest"]) + sBullet \
           + "K-Nearest Neighbors --> " + str(accuracies["K-Nearest Neighbors"])
    messagebox.showinfo("Summary Results", text)


def browse(fileBrowsing, root):
    filename = filedialog.askopenfilename()
    fileBrowsing.insert(0, filename)
    root.update()
    #initialState(FILE_NAME)

def mainWithGui():
    isSummary = False
    plot.rcParams["figure.figsize"] = (15, 6)
    accuracies = {}
    root = Tk()
    root.title("IMDB Scores Prediction")
    canvas1 = tk.Canvas(root, width=800, height=800, bd=0, highlightthickness=0)
    canvas1.configure(background="#82E0D9")
    canvas1.pack()

    imagepath1 = r'/Users/pabloaramburu/PycharmProjects/IMDB-Pablo/a1.jpg'  # include the path for the image (use 'r' before the path string to address any special character such as '\'. Also, make sure to include the image type - here it is jpg)
    image = Image.open(imagepath1)
    # image = image.filter(ImageFilter.BLUR)
    image = image.resize((1200, 500), Image.ANTIALIAS)
    image1 = ImageTk.PhotoImage(image)  # PIL module
    canvas1.create_image(200, 100, image=image1)

    ''' SUPERFRAME CONTINING TITLE'''
    superFrame1 = tk.Frame(root, bg="#82E0D9", bd=5)
    superFrame1.place(relx=0.01, rely=0.54, relwidth=0.97, relheight=0.25)
    label1 = Label(superFrame1, bg="#82E0D9",
                   text="Welcome to the IMDB Scores Prediction. Please, choose a .csv file.",
                   font="Helvetica 20 bold", fg='black')
    label1.pack()

    ''' SUPERFRAME CONTAINIGN BROWSE'''
    border = tk.Frame(root, bg="black", bd=5)
    border.place(relx=0.047, rely=0.598, relwidth=0.905, relheight=0.065)
    superFrame3 = tk.Frame(root, bg="white", bd=5)
    superFrame3.place(relx=0.05, rely=0.6, relwidth=0.9, relheight=0.06)
    fileBrowsing = tk.Entry(superFrame3)
    browseButton = Button(superFrame3, text="Browse", command=lambda: browse(fileBrowsing, root))
    fileBrowsing.place(width=600)
    browseButton.place(x=601, width=100)

    while True:
        if fileBrowsing.get()!= '':
            if fileBrowsing.get()[-3:] == 'csv':
                data = initialState(FILE_NAME)
                dataKNN = getDataKNN(FILE_NAME)
                messagebox.showinfo('Success', 'You entered a valid csv file. Options will be shown below')
                break
            else:
                fileBrowsing.delete(0, tk.END)
                fileBrowsing.insert(0,'')
                messagebox.showwarning('Error', 'You must enter a valid .csv file')
        root.update()

    ''' SUPERFRAME CONTAINING SUBFRAMES OF BUTTONS'''
    superFrame2 = tk.Frame(root, bg="#82E0D9", bd=5)
    superFrame2.place(relx=0.01, rely=0.69, relwidth=0.97, relheight=0.25)

    ''' FRAME 2 CONTAINING DATA EXPLORATION'''
    # frame1b is the border, the other contains the buttons
    frame2b = tk.Frame(superFrame2, bg="black", bd=10)
    frame2b.place(relx=0.19, rely=0.04, relwidth=0.305, relheight=0.875, anchor='n')
    frame2 = tk.Frame(superFrame2, bg="white", bd=10)
    frame2.place(relx=0.19, rely=0.05, relwidth=0.3, relheight=0.85, anchor='n')
    label1 = Label(frame2, text="Data Exploration", font="Helvetica 16 bold", fg='#C3A278')
    label1.place(relwidth=0.8, relx=0.1)
    buttonExploreCountry = Button(frame2, text="Analysis per country", command=lambda: analysisPerCountry(data))
    buttonExploreCountry.place(relwidth=0.8, relx=0.1, rely=0.5)
    buttonExploreGenre = Button(frame2, text="Analysis per genre", command=lambda: analysisPerGenre(data))
    buttonExploreGenre.place(relwidth=0.8, relx=0.1, rely=0.75)
    buttonCorr = Button(frame2, text="Correlation", command=lambda : correlation(data))
    buttonCorr.place(relwidth=0.8, relx=0.1, rely=0.25)
    # canvas1.create_window(375, 400, window=buttonExploreData)

    ''' FRAME 3 CONTAINING PREDICTION METHODS'''
    frame3b = tk.Frame(superFrame2, bg="black", bd=10)
    frame3b.place(relx=0.70, rely=0, relwidth=0.505, relheight=1.0, anchor='n')
    frame3 = tk.Frame(superFrame2, bg="white", bd=10)
    frame3.place(relx=0.7, rely=0.01, relwidth=0.5, relheight=0.98, anchor='n')
    label2 = Label(frame3, text="Prediction algorithms", font="Helvetica 16 bold", fg='#C3A278')
    label2.place(relwidth=0.8, relx=0.1)
    buttonLR = Button(frame3, text="Logistic Regression",
                      command=lambda: linearRegressionModel(data, accuracies, isSummary))
    buttonLR.place(relwidth=0.45, relx=0.01, rely=0.2)
    buttonNBG = Button(frame3, text="Naives Bayes Gaussian",
                       command=lambda: naiveBayesGaussian(data, accuracies, isSummary))
    buttonNBG.place(relwidth=0.45, relx=0.01, rely=0.4)
    buttonNBB = Button(frame3, text="Naives Bayes Bernoulli",
                       command=lambda: naiveBayesBernoulli(data, accuracies, isSummary))
    buttonNBB.place(relwidth=0.45, relx=0.01, rely=0.6)
    buttonSVM = Button(frame3, text="Support Vector Machines",
                       command=lambda: supportVectorMachines(data, accuracies, isSummary))
    buttonSVM.place(relwidth=0.50, relx=0.51, rely=0.2)
    buttonRF = Button(frame3, text="Random Forest", command=lambda: randomForest(data, accuracies, isSummary))
    buttonRF.place(relwidth=0.45, relx=0.535, rely=0.4)
    buttonKNN = Button(frame3, text="K-Nearest Neighbors",
                       command=lambda: kNearestNeighbors(dataKNN, accuracies, isSummary))
    buttonKNN.place(relwidth=0.45, relx=0.535, rely=0.6)
    buttonSumm = Button(frame3, text="Summary of algorithms",
                        command=lambda: summary(data, dataKNN, accuracies, isSummary))
    buttonSumm.place(relwidth=0.5, relx=0.24, rely=0.84)

    root.mainloop()


if __name__ == '__main__':
    #main()
    mainWithGui()

# https://stackoverflow.com/questions/42579927/rounded-button-tkinter-python
