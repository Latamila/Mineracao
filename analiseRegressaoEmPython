#BAIXO OS PACOTES MESMO QUE ACREDITE QUE JÁ OS TENHA.
!pip install pandas
!pip install numpy
!pip install scipy
!pip install sklearn
!pip install matplotlib

#IMPORTAR APRENDIZADO DE MAQUINA

import numpy as np
import panda as pd
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error\n
from sklearn.preprocessing import StandardScaler\n
from sklearn.model_selection import train_test_split #nao consegui validar com 'sklearn.cross_validation', por isso o buscador me trouxe essa solução.

df= pd.read_csv("C:/Users/Camila_Data_Science/Downloads/winequality-red.csv")
df.head()

x=df.loc[:, : "alcohol"] #salvar as variáveis preditoras e resposta (outcomes)
y=df["quality"]          #em objetos separados

TRANSFORMAÇÃO DE DADOS

Escalar as variáveis antes de modelar.

scaler= StandardScaler()
x= scaler.fit_transform(x)

DIVIDIR OS DADOS EM DOIS

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.333, random_state=1)

ANÁLISE

CRIAR MODELO DE REGRESSÃO LASSO

lasso = Lasso(alpha=0.001)
lasso.fit(x_train, y_train)

SVR= REGRESSAO POR VETOR DE SUPORTE

svr= SVR(C=8, epsilon=0.2, gamma=0.5)
svr.fit(x_train, y_train)

SVR(C=8, epsilon=0.2, gamma=0.5)
y_pred_lasso= np.round (np.clip(lasso.predict(x_test), 1, 10)).astype(int) 
np.round (1 - mean_squared_error(y_test, y_pred_lasso)/y_test.std(), 2)

y_pred_svr= np.round (np.clip(lasso.predict(x_test), 1, 10)).astype(int) 
np.round (1 - mean_squared_error(y_test, y_pred_svr)/y_test.std(), 2)

