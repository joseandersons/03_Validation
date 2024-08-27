import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import requests
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

data = pd.read_excel('abalone_dataset.xlsx')

# Codificar o atributo categórico 'Sex' (M -> 0, F -> 1, I -> 2)
label_encoder = LabelEncoder()
data['sex'] = label_encoder.fit_transform(data['sex'])

#columns        sexo	comprimento	  diametro	  altura    peso_total	peso_descascado	 peso_visceras	   peso_casca      tipo
#columns        sex	    length	  diameter	  height    whole_weight	shucked_weight	 viscera_weight	   shell_weight	   type
feature_cols = ['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight']

X = data[feature_cols]
y = data.type


# Dividir os dados em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar o modelo KNN
neigh = KNeighborsClassifier(n_neighbors=3) # 3 vizinhos mais próximos    #KNN
model = RandomForestClassifier(n_estimators=100, random_state=42)         #RandomForest
model_svm = SVC(kernel='rbf', random_state=42)
model_logreg = LogisticRegression(random_state=42)                            #Svm

# Treinar o modelo
neigh.fit(X_train, y_train)
model.fit(X_train, y_train)
model_logreg.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred_test_knn = neigh.predict(X_test)
y_pred_test_random_forest = model.predict(X_test)
model_svm.fit(X_train, y_train)
y_pred_test_logreg = model_logreg.predict(X_test)

# Avaliar o modelo usando acurácia
accuracy_knn = accuracy_score(y_test, y_pred_test_knn)
accuracy_random_forest = accuracy_score(y_test, y_pred_test_random_forest)
y_pred_test_svm = model_svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_test_svm)
accuracy_logreg = accuracy_score(y_test, y_pred_test_logreg)

print(f'Acurácia no conjunto de teste alg knn: {accuracy_knn:.4f}')
print(f'Acurácia no conjunto de teste alg random forest: {accuracy_random_forest:.4f}')
print(f'Acurácia no conjunto de teste alg SVM: {accuracy_svm:.4f}')
print(f'Acurácia no conjunto de teste alg Logistic Regression: {accuracy_logreg:.4f}')













#####Envio para o Servidor
# # Enviando previsões realizadas com o modelo para o servidor
# URL = "https://aydanomachado.com/mlclass/03_Validation.php"

# #TODO Substituir pela sua chave aqui
# DEV_KEY = "Careless Whisper"

# # json para ser enviado para o servidor
# data = {'dev_key':DEV_KEY,
#         'predictions':pd.Series(y_pred).to_json(orient='values')}

# # Enviando requisição e salvando o objeto resposta
# r = requests.post(url = URL, data = data)

# # Extraindo e imprimindo o texto da resposta
# pastebin_url = r.text
# print(" - Resposta do servidor:\n", r.text, "\n")