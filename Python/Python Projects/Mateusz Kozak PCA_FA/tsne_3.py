import pandas as pd
import matplotlib.pyplot as plt
from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


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


from sklearn.manifold import TSNE
import seaborn as sb

data_factors6 = TSNE(n_components=2).fit_transform(scaled_data)
#data_transformed = data_factors.fit_transform(scaled_data)
print(data_factors6)

columns = dataset.columns
print(columns)

#paletka = sb.palplot(sb.color_palette("RdBu", n_colors=7))
sb.set_color_codes("dark")
##Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, icefire, icefire_r, inferno, inferno_r, jet, jet_r, magma, magma_r, mako, mako_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, rocket, rocket_r, seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, viridis, viridis_r, vlag, vlag_r, winter, winter_r

#plt.figure(figsize=(20,20))
#
plt.figure(figsize=(15,10))
sb.scatterplot(x=data_factors6.T[0], y=data_factors6.T[1], data=dataset, hue=columns[9],
               size=columns[-3], style=columns[3], sizes=(10,400)
               , palette="RdBu_r")
#
#plt.show()
#plt.subplot(3,1,2)
plt.figure(figsize=(15,10))
sb.scatterplot(x=data_factors6.T[0], y=data_factors6.T[1], data=dataset, hue=columns[9],
               size=columns[7], style=columns[-2], sizes=(10,400),
               palette="gist_rainbow_r")
#plt.show()
#plt.subplot(3,1,3)
plt.figure(figsize=(15,10))
sb.scatterplot(x=data_factors6.T[0], y=data_factors6.T[1], data=dataset, hue=columns[7],
               size=columns[0], style=columns[3], sizes=(10,400),
               palette="magma_r")
plt.show()

plt.figure(figsize=(15,10))
sb.scatterplot(x=data_factors6.T[0], y=data_factors6.T[1], data=dataset, hue=columns[9],
               size=columns[-3], style=columns[3], sizes=(10,400),
            palette="RdBu_r", legend=None)
#plt.show()

#plt.show()
#plt.subplot(3,1,2)
plt.figure(figsize=(15,10))
sb.scatterplot(x=data_factors6.T[0], y=data_factors6.T[1], data=dataset, hue=columns[9],
               size=columns[7], style=columns[-2], sizes=(10,400),
               palette="gist_rainbow_r", legend=None)
#plt.show()

plt.figure(figsize=(15,10))
sb.scatterplot(x=data_factors6.T[0], y=data_factors6.T[1], data=dataset, hue=columns[7],
               size=columns[0], style=columns[3], sizes=(10,400),
               palette="magma_r", legend=None)
plt.show()