import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix, recall_score, precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV

df = pd.read_csv(r"income_evaluation.csv")
sort_names = set(df[" native-country"])
countries = []
for i in sort_names:
    countries.append(i)

Asian =[' Cambodia', ' China', ' Hong', ' India', ' Iran', ' Japan', 'Laos',
        ' Philippines', ' South', ' Taiwan', ' Thailand',  ' Vietnam']
North_America = [' Canada', ' United-States']
South_America = [' Columbia', ' Ecuador', ' El-Salvador', ' Guatemala', ' Honduras', ' Mexico', ' Nicaragua', ' Peru', ' Puerto-Rico']
Paccific =[' Cuba', ' Dominican-Republic', ' Haiti' , ' Jamaica', ' Trinadad&Tobago']
Eastern_Europe = [' Hungary', ' Poland', ' Yugoslavia']
West_Europe = [' England', ' France', ' Germany', ' Greece',
               ' Holand-Netherlands', ' Irland', ' Italy', ' Portugal', ' Scotland' ]

def continent_country_splitting(value):
    if value in Asian:
        return "Asia"
    elif value in North_America:
        return "North America"
    elif value in South_America:
        return " South America"
    elif value in Paccific:
        return " Paciffic"
    elif value in Eastern_Europe:
        return " Eastern Europe"
    elif value in West_Europe:
        return " Western Europe"
    else:
        return "Rest"

def hours_transformer(value):
    if value <40:
        return "40h<"
    elif value == 40:
        return "40h"
    elif value > 40:
        return "40h>"

df["hours_weekly_transofmed"] = df[" hours-per-week"].apply(hours_transformer)
df["continets"] = df[" native-country"].apply(continent_country_splitting)

data_set = df.drop(columns=[' fnlwgt',
                            ' capital-gain',
                            ' capital-loss',
                            ' hours-per-week',
                            ' education',
                            ' native-country',
                            " income"])

numeric_columns = data_set.select_dtypes(include =[np.number]).columns.tolist()
cat_columns = data_set.select_dtypes(exclude=[np.number]).columns.tolist()

## Pipline dataselector
class DataframeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, atrib_names):
        self.atribute_names = atrib_names
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=0):
        return X[self.atribute_names].values

class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = MultiLabelBinarizer(*args, **kwargs)
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)

## Data Pipelines construction
#numeric Pipeline
num_pipeline = Pipeline([
    ('selector', DataframeSelector(numeric_columns)),
    ('std_scaler', StandardScaler())
])
#categorial pipeline
cat_pipeline = Pipeline([
    ('selector', DataframeSelector(cat_columns)),
    ('label_binarizer', MyLabelBinarizer())
])
##union piplines
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipline", cat_pipeline)
])

y_dumies = pd.get_dummies(df[" income"])

X = data_set
y = y_dumies[" >50K"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

X_train_transfomed = full_pipeline.fit_transform(X_train)
X_test_transformed = full_pipeline.fit_transform(X_test)

model_logistic = LogisticRegression(solver="sag", max_iter=10000)
model_logistic.fit(X_train_transfomed,
                   np.ravel(y_train))
print(model_logistic.score(X_test_transformed,
                           np.ravel(y_test)))
#model_logistic.predict()
logistic_cross_val = cross_val_score(model_logistic, X_train_transfomed, np.ravel(y_train), cv=10 )

print(f"Cross validation logistic regresion: {np.mean(logistic_cross_val)}")
##tune the model
"""params = {"C": [0.1,0.2,0.5,1,2,10],
          "solver": ["newton-cg", "lbfgs", "liblinear", "sag", 'saga']
          }

grid_search = GridSearchCV(model_logistic, params, cv=3)
grid_search.fit(X_train_transfomed, np.array(y_train).reshape(-1,1))
print(grid_search.best_params_)
print(grid_search.best_score_)"""

model_tuned = LogisticRegression(C=1, solver="newton-cg", random_state=2)
model_logistic_tuned = model_tuned.fit(X_train_transfomed,
                   np.ravel(y_train))
print(model_tuned.score(X_test_transformed,
                           np.ravel(y_test)))

prediction = model_tuned.predict(X_test_transformed)
true_values = y_test
conf_matrix = confusion_matrix(true_values, prediction)
print(conf_matrix)
##metrics of Logistic regresion

precision = conf_matrix[0][0]/sum(conf_matrix[0])
recall = conf_matrix[1][1]/sum(conf_matrix[1])
recall_precison = [recall, precision]
print(recall_precison)

print(prediction)

## drzewo decyzyjne

from sklearn.tree import DecisionTreeClassifier

model_tree = DecisionTreeClassifier(random_state=2)
model_tree.fit(X_train_transfomed, np.ravel(y_train))
print(model_tree.score(X_train_transfomed, np.ravel(y_train)))

cross_vall_tree = cross_val_score(model_tree, X_train_transfomed,
                np.ravel(y_train), cv=10)
print(f"Cross validation Decision Tree: {np.mean(cross_vall_tree)}")
print(f"Tree depth : {model_tree.get_depth()}")
print(f"Leafs : {model_tree.get_n_leaves()}")
## tuning Decision Tree
params_tree = {"max_features": ["auto", "sqrt", "log2"],
               "max_depth": [1,2,3,4,5,7,8,10, 20,30,50],
               'min_samples_leaf': [1, 2, 4],
               'min_samples_split': [2,3,4, 5, 10]}

grid_serach_tree = GridSearchCV(model_tree, params_tree, cv=5)
grid_serach_tree.fit(X_train_transfomed,
                np.ravel(y_train))
print(f"Best params: {grid_serach_tree.best_params_}")
print(f"Best Score: {grid_serach_tree.best_score_}")

best_params_tree =  grid_serach_tree.best_params_

tuned_model_tree = DecisionTreeClassifier(random_state=2,
                                          max_depth=best_params_tree["max_depth"],
                                          min_samples_leaf=best_params_tree['min_samples_leaf'],
                                          max_features=best_params_tree["max_features"],
                                          min_samples_split=best_params_tree["min_samples_split"])

tuned_model_tree.fit(X_train_transfomed, np.ravel(y_train))
cross_vall_tree = cross_val_score(tuned_model_tree, X_train_transfomed,
                np.ravel(y_train), cv=10)
print(f"Cross validation Decision Tree: {np.mean(cross_vall_tree)}")
print(f"Tree depth : {model_tree.get_depth()}")
print(f"Leafs : {model_tree.get_n_leaves()}")
print(f"Model train score: {tuned_model_tree.score(X_train_transfomed, np.ravel(y_train))}")

##decision tree predictions
tree_predict = tuned_model_tree.predict(X_test_transformed)
true_tree = y_test

conf_matrix = confusion_matrix(true_tree, tree_predict)
print(conf_matrix)
##metrics of Logistic regresion
precision_tree = conf_matrix[0][0]/sum(conf_matrix[0])
recall_tree = conf_matrix[1][1]/sum(conf_matrix[1])

##Random Forest

from sklearn.ensemble import RandomForestClassifier

random_model = RandomForestClassifier(random_state=2, n_estimators=1000, max_depth=10)
model_tree.fit(X_train_transfomed, np.ravel(y_train))
print(f"Random forest score: {model_tree.score(X_test_transformed, np.ravel(y_test))}")

params_random = {'criterion' : ["gini", "entropy"],
                'max_depth': [5,10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                'min_samples_leaf': [2, 4, 6],
                'min_samples_split': [5, 10, 15],
                'n_estimators': [200, 400, 600, 800],
                'bootstrap': [True, False],
                'max_features': ['auto', 'sqrt', "log2"]}
#, 1000, 1200, 1400, 1600, 1800, 2000]}
#'max_features': ['auto', 'sqrt'],
#'bootstrap': [True, False],
grid_serach_forest = RandomizedSearchCV(random_model, params_random, cv=3)
grid_serach_forest.fit(X_train_transfomed, np.ravel(y_train))
forest_best_params = grid_serach_forest.best_params_
print(forest_best_params)

tuned_forest = RandomForestClassifier(random_state=2,
                                      criterion=forest_best_params["criterion"],
                                      bootstrap=forest_best_params["bootstrap"],
                                      max_depth=forest_best_params["max_depth"],
                                      max_features=forest_best_params["max_features"],
                                      min_samples_leaf=forest_best_params["min_samples_leaf"],
                                      min_samples_split=forest_best_params["min_samples_split"],
                                      n_estimators=forest_best_params["n_estimators"])

tuned_forest.fit(X_train_transfomed, np.ravel(y_train))
forest_predic = tuned_forest.predict(X_test_transformed)


conf_matrix = confusion_matrix(y_test, forest_predic)
print(conf_matrix)
##metrics of Logistic regresion
precision_forest = conf_matrix[0][0]/sum(conf_matrix[0])
recall_forest = conf_matrix[1][1]/sum(conf_matrix[1])


print(f"Tree Model score: {tuned_model_tree.score(X_test_transformed, np.ravel(y_test))}")
print(f"Tree model recall: {recall_tree}, recall score: {recall_score(y_test, tree_predict)}")
print(f"Tree model precision: {precision_tree},  precision_score: {precision_score(y_test, tree_predict)}")
print(f"Tree model roc_auc_score: {roc_auc_score(y_test, tree_predict)}")
#print(f"Tree model probability prediction: {tuned_model_tree.predict_proba(X_test_transformed)}")
print(f"Logistic model score: {model_tuned.score(X_test_transformed, np.ravel(y_test))}")
print(f"Logistic model recall: {recall}, , recall score: {recall_score(y_test,prediction)}")
print(f"Logistic model precision: {precision}, precision_score: {precision_score(y_test, prediction)}")
print(f"Logistic model roc_auc_score: {roc_auc_score(y_test, prediction)}")
#print(f"Logistic model probability prediction: {model_tuned.predict_proba(X_test_transformed)}")
print(f"Random forest model score: {tuned_forest.score(X_test_transformed, np.ravel(y_test))}")
print(f"Random forest recall: {recall_forest},recall score: {recall_score(y_test,forest_predic)}")
print(f"Random forest precision: {precision_forest},precision_score: {precision_score(y_test, forest_predic)}")
print(f"Random forest roc_auc_score: {roc_auc_score(y_test, forest_predic)}")

## Graphs

log_tp, log_fp, _ = roc_curve(y_test, prediction)
tree_tp, tree_fp, _ = roc_curve(y_test, tree_predict)
forest_tp, forest_fp, _ = roc_curve(y_test, forest_predic)
import matplotlib.pyplot as plt

plt.plot(log_tp, log_fp, c="r", label="Logistic model")
plt.plot(tree_tp, tree_fp, c="b", label= "Tree Model")
plt.plot(forest_tp, forest_fp, c="g", label= "Random Forest Model")
plt.ylabel("True positive")
plt.xlabel("False Positive")
plt.title("Model ROC curve comparison")
plt.legend()
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
