import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, f_classif

# importing data
shoot = pd.read_csv("shooting.csv", sep=",", encoding="ISO-8859-1").drop(["Sh", "SoT", "Matches"], axis=1)
passing = pd.read_csv("passing.csv", sep=",", encoding="ISO-8859-1").drop("Matches", axis=1)
possesion = pd.read_csv("possesion.csv", sep=",", encoding="ISO-8859-1").drop("Matches", axis=1)
pass_types = pd.read_csv("pass_types.csv", sep=",", encoding="ISO-8859-1").drop(["Matches", "In", "Out", "Str"], axis=1)
shoot_creation = pd.read_csv("shot_creation.csv",sep=",", encoding="ISO-8859-1").drop(['Matches', "SCA", "GCA"], axis=1)
deff = pd.read_csv("deffensive.csv", sep=",", encoding="ISO-8859-1").drop(["Matches", 'Tkl+Int', ], axis=1)
pass_types.rename(mapper={'Out.1': 'Out'}, axis=1, inplace=True)

# correcting players
players = [player.split("\\", 1)[1] for player in shoot.Player]
players = [player.replace("-", " ", 1) for player in players]

# temp = [player for player in players if "-" in player]
# for i in range(len(temp)):
#     print(temp[i], end=' ')
#     print()

# few of them are specific so I found with for loop them and corrected manually
players[players.index('Emile Smith-Rowe')] = 'Emile Smith Rowe'
players[players.index('Georges Kevin-NKoudou')] = 'Georges-Kevin NKoudou'
players[players.index('Alexis Mac-Allister')] = 'Alexis Mac Allister'
players[players.index('Giovani Lo-Celso')] = 'Giovani Lo Celso'
players[players.index('Anwar El-Ghazi')] = 'Anwar El Ghazi'
players[players.index('David de-Gea')] = 'David de Gea'
players[players.index('Jean Philippe-Gbamin')] = 'Jean-Philippe Gbamin'
players[players.index('Virgil van-Dijk')] = 'Virgil van Dijk'
players[players.index('Kevin De-Bruyne')] = 'Kevin De Bruyne'
players[players.index('Johann Berg-Gudmundsson')] = 'Johann Berg Gudmundsson'
players[players.index('Pierre Emerick-Aubameyang')] = 'Pierre-Emerick Aubameyang'
players[players.index('Patrick van-Aanholt')] = 'Patrick van Aanholt'

# setting appropiate names in every df
# earlier checked - every df had the same order of observations
# with the same players
stats = [shoot, passing, possesion, pass_types, shoot_creation, deff]
for df in stats:
    df.Player = players

# looking for duplicates - few players changed team during winter transfer window
# and data for them was stored in 2 observations, 1 for every team

# print(len(players) == len(set(players)))

# finding indexes of duplicated players
duplicated_inx = [i for i, x in enumerate(players) if players.count(x) > 1]

# print(shoot.iloc[duplicated_inx])

# deleting duplicated rows
for df in stats:
    df.drop(duplicated_inx, inplace=True)
    df.reset_index(drop=True, inplace=True)

# print(shoot.shape)

# some of the players didn't play that much in 19/20 season
# decided to delete those who has played less than equivalents of 8 macthes
drop_mins = [i for i, x in enumerate(shoot['90s']) if x < 3]

# print(len(drop_mins))

# its not a great deal to delete another observations, but these players played less than
# 270 minutes that season, so I believe that's not a big loss of information
for df in stats:
    df.drop(drop_mins, inplace=True)
    df.reset_index(drop=True, inplace=True)

# choosing only main position for every player
# some of them played a part of games on different positions
positions = [position[:2] for position in shoot.Pos]
for df in stats:
    df.Pos = positions

# checking NAs
# for df in stats:
#     print(df.isnull().sum())

# 1 NA in Age and Born columns but these cols will be deleted - won't be necessary for classification
# moreover
# i can see that all other missing values appeared because of dividing by 0
# for example (most or probably all of) GKs, as their role is to save shots, didn't take any shot
# so Goals scored / Shot On Target cannot be calculated.
# I decided to fill the gaps with 0, as it should be ok in case of classifying player with their pitch roles
# Age and Born column will also be filled with 0 for Luke Thomas, but these columns will be deleted
# from other than shoot df, before joining df
for df in stats:
    df.fillna(0, inplace=True)

# print("After filling gaps")
# for df in stats:
#     print(df.isnull().sum())


# deleting cols that won't be necessary for classification task or are repeated columns from data frames before merging
del stats[0]
for df in stats:
    df.drop(['Player', 'Nation', 'Pos', 'Squad', 'Age', 'Born', '90s'], axis=1, inplace=True)
shoot.drop(['Nation', 'Squad', 'Age', 'Born', '90s'], axis=1, inplace=True)

data = shoot
for df in stats:
    data = pd.merge(data, df, on='Rk')

# print(data.head())
# print(data.info())
# print(data.isnull().sum().values)

# setting player names as index of data
data.set_index('Player', inplace=True)

# data has more than 100 features - it's too much
# dimensionality reduction with PCA

x = data.drop(["Pos", "Rk"], axis=1)
y = data.Pos

theme_bw = "theme_bw.mplstyle"
pca = PCA().fit(x)
plt.plot(list(range(1, 11)), np.cumsum(pca.explained_variance_ratio_[:10]), marker='o', linestyle='--', color='black')
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.style.use(theme_bw)
plt.suptitle("PCA results")
plt.show()

# 3 new dimensions seems to be reasonable - it explains above 98% variance of this data set
# choosing more dimensions shouldn't significantly improve results, cause next dimensions explains
# less than 0.04% of variance

pca = PCA(n_components=3)
x = pca.fit_transform(x)
x = preprocessing.scale(x)



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10, stratify=y)


KNN = KNeighborsClassifier(weights="distance")
param_grid_knn = {'n_neighbors': list(range(5, 121, 5))}

KNN_grid = GridSearchCV(estimator=KNN, param_grid=param_grid_knn,
                        scoring="accuracy", n_jobs=1, cv=5, return_train_score=True)

KNN_grid.fit(X_train, y_train)

KNN_pred = KNN_grid.best_estimator_.predict(X_test)
print("Best k in KNN:\n" + str(KNN_grid.best_params_))
KNN_acc = accuracy_score(y_test, KNN_pred)
print("KNN test set accuracy: " + str(round(KNN_acc, 3)))


# Create the parameter grid based on the results of random search
param_grid = {
    'max_depth': [1, 2, 3, 5],
    'max_features': [2, 3],
    'n_estimators': [50, 100, 250]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
RFgrid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                             cv=5, n_jobs=1)

RFgrid_search.fit(X_train, y_train)

RF_pred = RFgrid_search.best_estimator_.predict(X_test)
print("Best params' combination:\n" + str(RFgrid_search.best_params_))
RF_acc = accuracy_score(y_test, RF_pred)
print("Random forest test set accuracy: " + str(round(RF_acc, 3)))

# comparison of the classifiers' trained on pca data results
# with feature selection based models (same as previous, with identical model building schema

x1 = data.drop(["Pos", "Rk"], axis=1)
y1 = data.Pos

# selecting 7 vars based on anova f-value, using SelectKBest
selector = SelectKBest(f_classif, k=7)
sel = selector.fit_transform(x1, y1)
variables = x1.columns[selector.get_support()].values

x1 = data[variables]

print("Result for models based on data containing 7 vars chosen via SelectKBest")

X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state=10, stratify=y1)


KNN = KNeighborsClassifier(weights="distance")
param_grid_knn = {'n_neighbors': list(range(5, 121, 5))}

KNN_grid = GridSearchCV(estimator=KNN, param_grid=param_grid_knn,
                        scoring="accuracy", n_jobs=1, cv=5, return_train_score=True)

KNN_grid.fit(X_train, y_train)

KNN_pred = KNN_grid.best_estimator_.predict(X_test)
print("Best k in KNN:\n" + str(KNN_grid.best_params_))
KNN_acc = accuracy_score(y_test, KNN_pred)
print("KNN test set accuracy: " + str(round(KNN_acc, 3)))


# Create the parameter grid based on the results of random search
param_grid = {
    'max_depth': [1, 2, 3, 5],
    'max_features': [2, 3],
    'n_estimators': [50, 100, 250]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
RFgrid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                             cv=5, n_jobs=1)

RFgrid_search.fit(X_train, y_train)

RF_pred = RFgrid_search.best_estimator_.predict(X_test)
print("Best params' combination:\n" + str(RFgrid_search.best_params_))
RF_acc = accuracy_score(y_test, RF_pred)
print("Random forest test set accuracy: " + str(round(RF_acc, 3)))


# even though results shows that models based on 'normal' data was better
# than models based on 'pca' data, the difference was not so huge

# at the end, there is cool way to plot 'pca data' with 3D plot
# showing that pca may be useful tool in case on visualization
from mpl_toolkits.mplot3d.axes3d import Axes3D

# PCA - 3 wymiary
x_reduced = PCA(n_components=3).fit_transform(x)
fig = plt.figure()
ax = Axes3D(fig)
ax.set_title('Soccer Dataset by PCA', size=14)
gk_ind = y == "GK"
ax.scatter(x_reduced[gk_ind, 0], x_reduced[gk_ind, 1], x_reduced[gk_ind, 2], c="black", label="Goalkeepers")
df_ind = y == "DF"
ax.scatter(x_reduced[df_ind, 0], x_reduced[df_ind, 1], x_reduced[df_ind, 2], c="red", label="Defenders")
mf_ind = y == "MF"
ax.scatter(x_reduced[mf_ind, 0], x_reduced[mf_ind, 1], x_reduced[mf_ind, 2], c="blue", label="Midfielders")
fw_ind = y == "FW"
ax.scatter(x_reduced[fw_ind, 0], x_reduced[fw_ind, 1], x_reduced[fw_ind, 2], c="yellow", label="Forwards")
ax.set_xlabel('First dimension')
ax.set_ylabel('Second dimension')
ax.set_zlabel('Third dimension')
ax.legend()
plt.style.use(theme_bw)
plt.show()