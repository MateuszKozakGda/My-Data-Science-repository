import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import statsmodels.api as sm

df = pd.read_csv(r"Salary_Data.csv")

X = df["YearsExperience"]
y = df["Salary"]

print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=2)

X_test_reshape = np.array(X_test).reshape(-1,1)
X_train_reshape = np.array(X_train).reshape(-1,1)
Y_test_reshape = np.array(y_test).reshape(-1,1)
Y_train_reshape = np.array(y_train).reshape(-1,1)


model = LinearRegression()
scores = model.fit(X_train_reshape, Y_train_reshape)

cross_val = cross_val_score(model, X_train_reshape, Y_train_reshape, scoring="neg_mean_squared_error", cv=3)
cross_val_wynik = np.sqrt(-cross_val)

#X_predykcja = scores()
params = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
grid_search =GridSearchCV(model, params, cv=None)
grid_search.fit(X_train_reshape, Y_train_reshape)
print ("r2 / variance : ", grid_search.best_score_)
print("Best prameters: ", grid_search.best_params_)
print("Residual sum of squares: %.2f"
              % np.mean((grid_search.predict(X_test_reshape) - Y_test_reshape) ** 2))

predicts = scores.predict(X_test_reshape)

all_data_predictions = scores.predict(np.array(X).reshape(-1,1))

def RSE(prediction, true):
    num_samples = len(true)
    delta = true-predicts
    return delta

def display_scores(scores):
    print(f"Wynik: {scores}")
    print(f"Srednia: {scores.mean()}")
    print(f"Odchylenie standardowe: {scores.std()}")
display_scores(cross_val_wynik)
print(f"Współczynnik R^2 dla zbioru testowego: {scores.score(X_train_reshape, Y_train_reshape)}")
print(f"MSE: {mean_squared_error(y, all_data_predictions)}")
print(f"RSE : {np.sqrt((1/(len(y)-2)*mean_squared_error(y, all_data_predictions)))}")

import seaborn as sb


plt.figure()
plt.scatter(X,y, label="True Data")
plt.plot(X, all_data_predictions, c="r", label="Prediction")

plt.xlabel("Years of Experiance")
plt.ylabel("Sallary")
plt.title("Dependance of Sallary to Years of Experiance")
plt.legend(loc ="lower right")
plt.show()

#x_reg, y_reg = pd.Series(X, name="x_reg"), pd.Series(all_data_predictions, name="predictions")
#print(np.shape(X), np.shape(np.ravel(all_data_predictions)))
sb.regplot(X, y ,marker="+", ci=95)
plt.title("Coinfidence 95%")
plt.show()