import numpy as np 
import pandas as pd
import statistics
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import shapiro, ttest_ind, f_oneway
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, r2_score, mean_squared_error
np.random.seed(42)

# Nos données 
scaler = MinMaxScaler()
data = pd.DataFrame(
    {
        'poids': np.random.normal(2.5,0.5,100),
        'nourriture': np.random.normal(1.2,0.3,100),
        'température': np.random.normal(25,2,100)
    }
)

print(data.head)
filename = "donnes_generees_poulet.csv"
data.to_csv(filename)

scaled_poids = scaler.fit_transform(data[['poids']])
scaled_nourriture = scaler.fit_transform(data[['nourriture']])
scaled_temperature = scaler.fit_transform(data[['température']])

# Exercice 1 : Statistiques descriptives
# Moyenne
moy1 = np.mean(data['poids'])
print(f"La moyenne du poids est de : {moy1:.2f}")
moy2 = np.mean(data['nourriture'])
print(f"La moyenne de la nourriture est de : {moy2:.2f}")
moy3 = np.mean(data['température'])
print(f"La moyenne de la température est de : {moy3:.2f}")
# Médiane
med1 = np.median(data['poids'])
print(f"La médiane du poids est de : {med1:.2f}")
med2 = np.median(data['nourriture'])
print(f"La médiane du poids est de : {med2:.2f}")
med3 = np.median(data['température'])
print(f"La médiane du poids est de : {med3:.2f}")
# Ecart-type
std1 = np.std(data['poids'])
print(f"La déviation standard du poids est de : {std1:.2f}")
std2 = np.std(data['nourriture'])
print(f"La déviation standard de la nourriture est de : {std2:.2f}")
std3 = np.std(data['température'])
print(f"La déviation standard de la température est de : {std3:.2f}")
# Variance
var1 = statistics.variance(data['poids'])
print(f"La variance du poids est de : {var1:.2f}")
var2 = statistics.variance(data['nourriture'])
print(f"La variance de la nourriture est de : {var2:.2f}")
var3 = statistics.variance(data['température'])
print(f"La variance de la température est de : {var3:.2f}")
# Les-Quartiles
qart1 = data['poids'].quantile(0.25)
qart2 = data['poids'].quantile(0.5)
qart3 = data['poids'].quantile(0.75)

print(f"Les quartiles du poids sont : 25% : {qart1:.2f} , 50% {qart2:.2f} : , 75% : {qart3:.2f}")
qart1_1 = data['nourriture'].quantile(0.25)
qart1_2 = data['nourriture'].quantile(0.5)
qart1_3 = data['nourriture'].quantile(0.75)
print(f"Les quartiles de la nourriture sont : 25% : {qart1_1:.2f} , 50% {qart1_2:.2f} : , 75% : {qart1_3:.2f}")

qart2_1 = data['température'].quantile(0.25)
qart2_2 = data['température'].quantile(0.5)
qart2_3 = data['température'].quantile(0.75)
print(f"Les quartiles de la température sont : 25% : {qart2_1:.2f} , 50% {qart2_2:.2f} : , 75% : {qart2_3:.2f}")

# Visualisation
n_bins = 10

plt.hist(data['poids'],bins=n_bins)
plt.title("Distribution du poids")
plt.show()
plt.hist(data['nourriture'],bins=n_bins)
plt.title("Distribution de la nourriture")
plt.show()
plt.hist(data['température'],bins=n_bins)
plt.title("Distribution de la température")
plt.show()

# IQR 
IQR_poids = qart3 - qart1
print(f"IQR du poids = {IQR_poids:.2f}")
v_aber_poids_inf = qart1 - 1.5  * IQR_poids
v_aber_poids_sup = qart3 + 1.5  * IQR_poids
print(f"Seuile de valeurs aberrantes poids : inférieur à {v_aber_poids_inf:.2f} ou superieur à  {v_aber_poids_sup:.2f} ")
outliers = []
for k in data['poids']:
    if (k < v_aber_poids_inf or k > v_aber_poids_sup):
        outliers.append(k)
print(outliers)

IQR_nourriture = qart1_3 - qart1_1
print(f"IQR de la nourriture = {IQR_nourriture:.2f}")
v_aber_nourriture_inf = qart1_1 - 1.5  * IQR_nourriture
v_aber_nourriture_sup = qart1_3 + 1.5  * IQR_nourriture
print(f"Seuile de valeurs aberrantes nourriture : inférieur à {v_aber_nourriture_inf:.2f} ou superieur à  {v_aber_nourriture_sup:.2f} ")
outliers2 = []
for k in data['nourriture']:
    if (k < v_aber_nourriture_inf or k > v_aber_nourriture_sup):
        outliers2.append(k)
print(outliers2)

IQR_temperature = qart2_3 - qart2_1
print(f"IQR de la température = {IQR_temperature:.2f}")
v_aber_temperature_inf = qart2_1 - 1.5  * IQR_temperature
v_aber_temperature_sup = qart2_3 + 1.5  * IQR_temperature
print(f"Seuile de valeurs aberrantes temperature : inférieur à {v_aber_temperature_inf:.2f} ou superieur à  {v_aber_temperature_sup:.2f} ")
outliers3 = []
for k in data['température']:
    if (k < v_aber_temperature_inf or k > v_aber_temperature_sup):
        outliers3.append(k)
print(outliers3)
# Z_score

z_score_poids = stats.zscore(data['poids']) 
for z in z_score_poids:
    print(f"Z_score du poids = {z:.2f} ")
z_score_nourriture = stats.zscore(data['nourriture'])
for z in z_score_poids:
    print(f"Z_score de la nourriture = {z:.2f} ")
z_score_température = stats.zscore(data['température'])
for z in z_score_poids:
    print(f"Z_score de la température = {z:.2f} ")

plt.boxplot(outliers)
plt.title("Outlier poids")
plt.show()
plt.boxplot(outliers2)
plt.title("Outlier nourriture")
plt.show()
plt.boxplot(outliers3)
plt.title("Outlier température")
plt.show()

#Exercice 3 
for col in data.columns:
    stat, p = shapiro(data[col])
    print(f"Test de Shapiro-Wilk : {col}")
    print(f"Stat {stat}, p-value = {p:.3f}")
    if p > 0.05:
        print("Distribution normale")
    else :
        print("Distribution non normale ")


#Test t de Student 
groupe1 = data['poids'][:50]
groupe2 = data['poids'][50:]
t_stat, t_p = ttest_ind(groupe1, groupe2)
print(f"Test t de Student (sur le poids) : p-value = {t_p:.3f}")

# Création des groupes pour la température
data['groupe_temp'] = pd.cut(data['température'], 
                             bins=[-np.inf, 24, 26, np.inf], 
                             labels=['Bas','Moyen','Haut'])

# ANOVA sur les groupes de température
groupe_bas = data[data['groupe_temp'] == 'Bas']['température']
groupe_moyen = data[data['groupe_temp'] == 'Moyen']['température']
groupe_haut = data[data['groupe_temp'] == 'Haut']['température']

f_stat, f_p = f_oneway(groupe_bas, groupe_moyen, groupe_haut)
print(f"ANOVA (groupes de température) : p-value = {f_p:.3f}")

# Partie 2

data = pd.read_csv("donnees_elevage_poulet.csv")
print(data.head)

# Standardisation des données
X = data.select_dtypes(include=np.number).values
X_std = (X - X.mean(axis=0)) / X.std(axis=0)

# Matrice de covariance
cov_matrix = np.cov(X_std.T)
print("Matrice de covariance :\n",cov_matrix)

# Calcul des valeurs/vecteurs propres
eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)

sorted_idx = np.argsort(eigen_values)[::-1]
eigen_values = eigen_values[sorted_idx]
eigen_vectors = eigen_vectors[:,sorted_idx]

print("\nValeurs propres :",eigen_values)
print("\nVecteurs propres :\n",eigen_vectors)

#Sélection des 2 premières composantes
components = eigen_vectors[:, :2]
projection = X_std.dot(components)

plt.figure(figsize=(10, 6))
plt.scatter(projection[:, 0],projection[:, 1],alpha=0.5)
plt.xlabel('Première Composante Principale (PC1)')
plt.ylabel('Deuxième Composante Principale (PC2)')
plt.title('Projection des données sur les deux premières CP')
plt.grid(True)
plt.show()

# Calcul de la variance expliquée
total_variance = sum(eigen_values)
explained_variance = eigen_values / total_variance
cumulative_variance = np.cumsum(explained_variance)

print("\nVariance expliquée par composante :",explained_variance)
print("Variance cumulative :",cumulative_variance)

X_std = StandardScaler().fit_transform(X)

kernels = ['linear', 'rbf', 'poly']
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

for i, kernel in enumerate(kernels):
    #KernelPCA
    kpca = KernelPCA(n_components=2,kernel=kernel,gamma=0.1,degree=3) 
    X_kpca = kpca.fit_transform(X_std)
    
    #Visualisation
    axes[i].scatter(X_kpca[:, 0], X_kpca[:, 1], alpha=0.5)
    axes[i].set_title(f'Kernel {kernel.capitalize()}')
    axes[i].grid(True)

plt.show()

# Création d'une cible binaire (survie > 90% = 1)
data['Survie_binaire'] = (data['Taux_survie_%'] > 90).astype(int)  

#Séparation features/cible
X = data.drop(['Taux_survie_%', 'Survie_binaire'], axis=1)
y = data['Survie_binaire']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))

#Importance des caractéristiques
importances = rf.feature_importances_
features_df = pd.DataFrame({'Feature': X.columns, 'Importance':importances})
print("\nImportance des variables:\n",features_df.sort_values('Importance',ascending=False))


# Bossting

# Préparation des données
X = data.drop('Gain_poids_jour_g', axis=1)
y = data['Gain_poids_jour_g']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# AdaBoost
ada = AdaBoostRegressor(n_estimators=100,random_state=42)
ada.fit(X_train,y_train)
y_pred_ada = ada.predict(X_test)
rmse_ada = np.sqrt(mean_squared_error(y_test, y_pred_ada)) 
print("\nAdaBoost - RMSE:",rmse_ada)
print("AdaBoost - R²:", r2_score(y_test,y_pred_ada))

# Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=100,random_state=42)
gb.fit(X_train,y_train)
y_pred_gb = gb.predict(X_test)
rmse_gb = np.sqrt(mean_squared_error(y_test,y_pred_gb)) 
print("\nGradient Boosting - RMSE:",rmse_gb)
print("Gradient Boosting - R²:", r2_score(y_test,y_pred_gb))