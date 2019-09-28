from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, recall_score, precision_score
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

### 0 - setosa, 1 - versicolor, 2- virginica
### features names : 0 - sepal lengh, 1- sepal width, 2 - petal lengh, petal - width

data = load_iris()
#print(data.feature_names)
X_data = data.data
Y_data = data.target

Y_data = pd.Series(Y_data)

feature_names = data.feature_names
target_names = data.target_names
target_list  = [name for name in target_names]
print(target_list)

def flower(num):
    if num == 0:
        return "setosa"
    if num == 1:
        return "versicolor"
    else:
        "virginica"

X_data = pd.DataFrame(X_data, columns=feature_names)

merged_data = X_data
merged_data["target"] = Y_data
#merged_data["target"] = merged_data["target"].apply(flower)

### Shuffle dataset
merged_data = merged_data.sample(frac=1, random_state=2).reset_index(drop=True)
print(merged_data[:3].head(5))

# draw correlation plot

sb.pairplot(X_data)
plt.show()
plt.figure(figsize=(20,20),tight_layout=True)
plt.matshow(X_data.corr(), cmap="seismic")
plt.xticks(range(len(X_data.columns)), X_data.columns, rotation=80)
plt.yticks(range(len(X_data.columns)), X_data.columns)
plt.colorbar()
plt.show()

##Splitting data_sets

column_names = list(merged_data.columns)
print(column_names)

X1_data = merged_data[column_names[:2]]
X2_data = merged_data[column_names[:3]]
X3_data = merged_data[column_names[:4]]
Y_data = merged_data[column_names[-1]]
print(X2_data.head())

list_of_X = {"2" : X1_data,
             "3" : X2_data,
             "4" : X3_data}
classifiers = {"K Neighbors Classifier": KNeighborsClassifier(),
               "Random Forest Classifier": RandomForestClassifier(random_state=2, n_estimators=100)}
##
print(f"shape X2: {np.shape(X2_data)}")
print(f"shpe Y: {np.shape(Y_data)}")
model_test = RandomForestClassifier()
model_multi = model_test.fit(X1_data, Y_data)
score = model_multi.score(X1_data, Y_data)
cv2 = cross_val_score(model_multi, X1_data, Y_data, cv=5)
print(score)
print(f"cv: {cv2.mean()}")
##True model construction

knn_results = []
rf_results = []


for key, classifier in classifiers.items():
    for key2, X in list_of_X.items():
        X_train, X_test, y_train, y_test = train_test_split(X, Y_data, test_size=0.1, random_state=2)

        scaller = MinMaxScaler()
        scaller.fit(X_train)
        X_train = scaller.transform(X_train)
        X_test = scaller.transform(X_test)

        model = classifier
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        prediction = model.predict(X_test)

        cv = cross_val_score(model, X_train, y_train, cv=5)
        conf_matrix = confusion_matrix(y_test, prediction)
        recall = recall_score(y_test, prediction, average="macro")
        prec=precision_score(y_test, prediction, average="macro")
        print(f"Actual training model: {key}")
        print(f"Number of features: {key2}")
        print(f"Model score: {score}")
        print(f"Cross val score: {cv.mean()}")
        print(f"Confusion matrix:\n {conf_matrix}")
        print(f"Recall: {recall}, precision: {prec}" )

        if key=="K Neighbors Classifier":
            knn_results.append(cv.mean())
        elif key=="Random Forest Classifier":
            rf_results.append(cv.mean())

#print(knn_results)
model_list = ["KNN", "Random Forest"]
labels_names = ["2 features", "3 features", "4 features"]

ind = np.arange(len(knn_results))  # the x locations for the groups
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
plot_knn = ax.bar(ind - width/2, knn_results, width, label=model_list[0])
plot_rfc = ax.bar(ind + width/2, rf_results, width, label=model_list[1])
ax.set_xticklabels(labels_names)
ax.set_xticks(range(0,3))
ax.set_ylabel("Model cross val score")
ax.set_title("Model scores")
ax.legend()

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = round(rect.get_height(),3)
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(3, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(plot_knn)
autolabel(plot_rfc)
fig.tight_layout()

plt.show()