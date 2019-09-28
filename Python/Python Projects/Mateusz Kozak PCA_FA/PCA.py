import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

dataset = pd.read_csv(r"C:\Users\Mateusz\Desktop\WA_Fn-UseC_-HR-Employee-Attrition_v2.csv", delimiter=',')
#print(dataset)

dataset2 = pd.get_dummies(dataset["Gender"])
dataset["GenderNumeric"] = dataset2["Male"]
dataset.head()

dataset.drop(["Age", "JobRole", "Gender",  "BusinessTravel", "Department", "EducationField", "MaritalStatus", "OverTime"], axis=1, inplace=True)
data_PCA = dataset
data_PCA.head()

scaler = StandardScaler()
scaler.fit(dataset)
scaled_data = scaler.transform(dataset)
print(scaled_data)

#data_factors = FactorAnalysis(n_components=19)
data_factors = PCA(n_components=6)
data_transformed = data_factors.fit_transform(scaled_data)

data_variance = data_factors.explained_variance_ratio_
print(data_variance)
ex_variance = np.var(data_transformed, axis=0)
ex_variance_ratio = ex_variance*100/np.sum(ex_variance)
iterator_list = 0
cumul_variance =0
while iterator_list<len(ex_variance_ratio):
    cumul_variance+=ex_variance_ratio[iterator_list]
    iterator_list+=1
print(cumul_variance)

#x = data_factors.get_covariance()
#print(x)

comp_labels =["PC"+str(i) for i in range(1, len(ex_variance_ratio)+1)]
print(comp_labels)

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
plt.matshow(data_factors.components_, cmap="seismic")
plt.yticks([i for i in range(len(comp_labels))], comp_labels, fontsize=10)
plt.colorbar()
plt.xticks(range(len(data_PCA.columns)), data_PCA.columns,
           rotation=65, ha="left")
plt.tight_layout()
plt.show()