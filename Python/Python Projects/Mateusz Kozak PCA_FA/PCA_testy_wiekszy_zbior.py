import pandas as pd
import matplotlib.pyplot as plt
from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA


# wczytanie pliku
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition_v2.csv")

encoder = LabelEncoder()
labels = df["EducationField"]
new_labels = encoder.fit_transform(labels)
print(new_labels)
df2 = df
df2["EducationField"] = new_labels
df["EducationField"]=df2["EducationField"]
labels_job = df["JobRole"]
new_job_labels = encoder.fit_transform(labels_job)
df2["job2"] = new_job_labels
df["JobRole"]=df2["job2"]



# zamiana wartości na numeryczne
df2 = pd.get_dummies(df["Gender"])
df["GenderNumeric"] = df2["Male"]
df2 = pd.get_dummies(df["OverTime"])
df["OverTimeNumeric"] = df2["Yes"]
#print(df3)

#print(df3)


# wgląd w dane i wybór zmiennych
df.info()

#print(df[["EducationField", "JobRole"]])
#print(df["EducationField"])
#print(df["JobRole"])
#print(df["Department"])
# usunięcie zmiennych nieliczbowych
df.drop(['BusinessTravel', 'Department',
         'MaritalStatus', 'OverTime', 'Gender'], axis=1, inplace=True)  # 'Age',

# Ocena zestawu danych - czy możemy znaleźć czynniki na podstawie zmiennych?

# Test sferyczności Bartletta sprawdza, czy obserwowane zmienne w ogóle korelują
# Wynik 0 dla p value oznacza, że test był istotny statystycznie, wskazując,
# że obserwowana macierz korelacji nie jest macierzą tożsamości.
chi_square_value, p_value = calculate_bartlett_sphericity(df)
print(f"chi_square : {chi_square_value} and p_Value:{p_value}")

# Test Kaiser-Meyer-Olkin (KMO) mierzy przydatność danych do analizy czynnikowej.
# Wartość KMO mniejsza niż 0,6 jest uważana za niewystarczającą.
kmo_all, kmo_model = calculate_kmo(df)
print(f"kmo_model: {kmo_model}")

dataset = df
data_PCA = dataset

scaler = StandardScaler()
scaler.fit(dataset)
scaled_data = scaler.transform(dataset)
#print(scaled_data)

#data_factors = FactorAnalysis(n_components=19)
components = 5
data_factors = PCA(n_components=components, svd_solver="randomized")
data_transformed = data_factors.fit_transform(scaled_data)

data_variance = data_factors.explained_variance_
#print(data_variance)
data_variance2 = data_factors.explained_variance_ratio_
print(f"variance ratio: {data_variance2}")
suma =0
a = 0
while a!=len(data_variance):
    suma += data_variance2[a]
    a+=1
print(f"suma wariancji: {suma}")
ex_variance = np.var(data_transformed, axis=0)
ex_variance_ratio = ex_variance*100/np.sum(ex_variance)
iterator_list = 0
cumul_variance =0
while iterator_list<len(ex_variance_ratio):
    cumul_variance+=ex_variance_ratio[iterator_list]
    iterator_list+=1
#print(cumul_variance)

#x = data_factors.get_covariance()
#print(x)

comp_labels =["PC"+str(i) for i in range(1, len(ex_variance_ratio)+1)]
#print(comp_labels)

plt.figure(figsize=(15,15))
plt.subplot(2,1,1)
plt.bar(x=range(1,len(ex_variance_ratio)+1), height=ex_variance_ratio, tick_label=comp_labels)
plt.ylabel("Percentage of variance")
plt.xlabel("Principal Component")
plt.title("Variance in terms of PC")

plt.subplot(2,1,2)
plt.plot(np.cumsum(data_variance))
plt.xlabel("Number of Components")
plt.ylabel("Variance Sum")
plt.title("Cumulative Variance")
plt.show()

#FA = FactorAnalysis(n_components=6)
#new_dataset_FA = FA.fit_transform(data_transformed)
#print(new_dataset_FA)

##corelation plot
plt.figure(figsize=(30,20))
plt.matshow(data_factors.components_, cmap="seismic")
plt.yticks([i for i in range(len(comp_labels))], comp_labels, fontsize=10)
plt.colorbar()
plt.xticks(range(len(data_PCA.columns)), data_PCA.columns,
           rotation=65, ha="left")
#plt.tight_layout()
plt.show()
### results data frame
results = pd.DataFrame(data_factors.components_.T, columns=comp_labels, index=data_PCA.columns)
print(results)
print(f"Data variance: {data_factors.explained_variance_}")
print(f"Varaince ratio: {data_factors.explained_variance_ratio_}")
print(f"Variance ratio cumsum: {np.cumsum(data_factors.explained_variance_ratio_)*100}")


data_to_loadings = [[data_factors.explained_variance_],
                    [data_factors.explained_variance_ratio_],
                    [np.cumsum(data_factors.explained_variance_ratio_)]]

#print(data_to_loadings)
#loading = pd.DataFrame(data_to_loadings ,columns=["PC"+str(i) for i in range(len(data_to_loadings[0]))],
                       #index=["Data variance", "Varaince ratio","Variance ratio cumsum"])
#print(loading)


results.to_csv(r"C:\Users\Mateusz\Desktop\PCA_DATA4.csv")

##scater matrix
#pd.plotting.scatter_matrix(data_PCA, alpha=0.3, diagonal="kde",figsize=(100,100))
#plt.show()
one = [1 for i in range(1,components+1)]
print(one)
x_labels = [x for x in range(0+1,components+1)]
plt.plot(range(1,components+1),data_variance)
plt.plot(range(1,components+1), one, "r--")
plt.xlabel("Number of components")
plt.ylabel("Variance")
#plt.xticks(range(0,components+1))
plt.show()


scaler2 = StandardScaler()
scaler2.fit(dataset)
scaled_data2 = scaler2.transform(dataset)
#print(scaled_data)

#data_factors = FactorAnalysis(n_components=19)
components2 = 2
data_factors2 = PCA(n_components=components2, svd_solver="randomized")
reduced_data = data_factors2.fit_transform(scaled_data2)

reduced_data = pd.DataFrame(reduced_data, columns=["PC1", "PC2"])
#print(reduced_data)

data_variance3 = data_factors2.explained_variance_
print(data_variance3)
data_variance4 = data_factors2.explained_variance_ratio_
print(f"variance ratio: {data_variance4}")



def biplot(data, reduced_data, pca):
    fig, ax = plt.subplots(figsize=(14, 8))

    # scatterplot of the reduced data
    ax.scatter(x=reduced_data.loc[:, 'PC1'], y=reduced_data.loc[:, 'PC2'], facecolors='b',
               edgecolors='b', s=70, alpha=0.5)

    feature_vectors = pca.components_.T

    # using scaling factors to make the arrows
    arrow_size, text_pos = 7.0, 8.0

    # projections of the original features
    for i, v in enumerate(feature_vectors):
        ax.arrow(0, 0, arrow_size * v[0], arrow_size * v[1], head_width=0.2, head_length=0.2, linewidth=2, color='red')
        ax.text(v[0] * text_pos, v[1] * text_pos, data.columns[i], color='black', ha='center', va='center',
                fontsize=8)

    ax.set_xlabel("PC1", fontsize=14)
    ax.set_ylabel("PC2", fontsize=14)
    ax.set_title("PC plane with original feature projections.", fontsize=16)
    return ax

biplot(dataset, reduced_data ,data_factors2)
plt.show()