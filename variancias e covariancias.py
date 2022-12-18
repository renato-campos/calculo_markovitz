import numpy as np
import pandas as pd
import statistics as stt

petr4_df = pd.read_csv('D:\\Projetos\\calculo_markovitz\\PETR4.SA.csv')
goau4_df = pd.read_csv('D:\\Projetos\\calculo_markovitz\\GOAU4.SA.csv')
usim5_df = pd.read_csv('D:\\Projetos\\calculo_markovitz\\USIM5.SA.csv')

petr4 = []
for i in range(1, len(petr4_df['Close'])):
    retorno = (petr4_df['Close'][i] - petr4_df['Close'][i-1]) / petr4_df['Close'][i-1]
    petr4.append(retorno)
del petr4_df

goau4 = []
for i in range(1, len(goau4_df['Close'])):
    retorno = (goau4_df['Close'][i] - goau4_df['Close'][i-1]) / goau4_df['Close'][i-1]
    goau4.append(retorno)
del goau4_df

usim5 = []
for i in range(1, len(usim5_df['Close'])):
    retorno = (usim5_df['Close'][i] - usim5_df['Close'][i-1]) / usim5_df['Close'][i-1]
    usim5.append(retorno)
del usim5_df

var_petr4 = stt.variance(petr4)
var_goau4 = stt.variance(goau4)
var_usim5 = stt.variance(usim5)

cov_petr_goau = stt.covariance(petr4, goau4)
cov_petr_usim = stt.covariance(petr4, usim5)
cov_goau_usim = stt.covariance(goau4, usim5)

print(f'''Variâncias:
var_PETR4 = {var_petr4:.6f}
var_GOAU4 = {var_goau4:.6f}
var_USIM5 = {var_usim5:.6f}
Covariâncias:
cov_PG = {cov_petr_goau:.6f}
cov_PU = {cov_petr_usim:.6f}
cov_GU = {cov_goau_usim:.6f}
''')

print(f'{2*var_petr4:.6f} {2*var_goau4:.6f} {2*var_usim5:.6f}')
print(f'{2*cov_petr_goau:.6f} {2*cov_petr_usim:.6f} {2*cov_goau_usim:.6f}')

A = np.array([[2*var_petr4, 2*cov_petr_goau, 2*cov_petr_usim, -1],
              [2*var_goau4, 2*cov_petr_goau, 2*cov_goau_usim, -1],
              [2*var_usim5, 2*cov_petr_usim, 2*cov_goau_usim, -1],
              [1, 1, 1, 0]])

B = np.array([[0],
              [0],
              [0],
              [1]])

A_inv = np.linalg.inv(A)
X = np.dot(A_inv, B)

print(f'{X[0][0]:.6f} , {X[1][0]:.6f} , {X[2][0]:.6f} , {X[3][0]:.6f}')
