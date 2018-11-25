import numpy as np
import statistics as st
from math import exp
import matplotlib.pyplot as plt




def datos_normalizar(me,ma,mi,value):
    data = (value-me)/(ma-mi)
    return data

def funcion_sigmoide(valor):
    s= 1/(1+exp(-valor))
    return s

valores_estadisticos = []
valores_entrenamiento = []
valores_prueba = []
entrada_entrenamiento = []
entrada_prueba = []
yn = []
xn = []
h = []
dataSet=np.loadtxt("datasetRegLog.txt",delimiter=';')
x=dataSet[:,[0,1]]
y=dataSet[:,2]

n =  int((np.size(dataSet))/len(dataSet[0]))

p60 = round(n*0.60)
p40 = n-p60
for i in range(0,p60):
    entrada_entrenamiento.append(x[i])
    valores_entrenamiento.append(y[i])
entrada_entrenamiento = np.array(entrada_entrenamiento).reshape(np.shape(entrada_entrenamiento))
valores_entrenamiento = np.array(valores_entrenamiento).reshape(np.shape(valores_entrenamiento))
maxy = max(valores_entrenamiento)
miny = min(valores_entrenamiento)
medy = st.mean(valores_entrenamiento)

for i in range(p60):
    yn.append(datos_normalizar(medy,maxy,miny,valores_entrenamiento[i]))
yn = np.array(yn).reshape(np.shape(valores_entrenamiento))

for i in range(len(entrada_entrenamiento[0])):
        maxi = max(entrada_entrenamiento[i])
        mini = min(entrada_entrenamiento[i])
        medi = st.mean(entrada_entrenamiento[i])
        valores_estadisticos.append([medi,mini,maxi])

for i in range (len(entrada_entrenamiento[0])):
    for j in range (p60):
        a = valores_estadisticos[i][0]
        b = valores_estadisticos[i][2]
        c = valores_estadisticos[i][1] 
        d = entrada_entrenamiento[j][i]
        xn.append(datos_normalizar(a,b,c,d))
xn = np.array(xn).reshape(np.shape(entrada_entrenamiento))
xn = np.c_[np.ones(p60),xn]
theta = [0,0,0]
it =10000

alpha = 0.00001
lista_sum = []
    
for j  in range(len(xn[0])):
    sumatoria = 0
    for i in range(p60):
        hi = theta[0]*xn[i][0] + theta[1]*xn[i][1] + theta[2]*xn[i][2]
        yi = yn[i] 
        sumatoria = sumatoria + (hi-yi)*xn[i][j]    
    lista_sum.append(sumatoria)
for i in range(len(xn[0])):
    temp = theta[i] - alpha*(1/p60)*lista_sum[i]
    theta[i] = temp
pass

yn = []
xn = []
for i in range(p60,n):
    entrada_prueba.append(x[i])
    valores_prueba.append(y[i])
entrada_prueba = np.array(entrada_prueba).reshape(np.shape(entrada_prueba))
valores_prueba = np.array(valores_prueba).reshape(np.shape(valores_prueba))

for i in range(p40):
    yn.append(datos_normalizar(medy,maxy,miny,valores_prueba[i]))
yn = np.array(yn).reshape(np.shape(valores_prueba))


for i in range (len(entrada_prueba[0])):
    for j in range (p40):
        a = valores_estadisticos[i][0]
        b = valores_estadisticos[i][2]
        c = valores_estadisticos[i][1] 
        d = entrada_entrenamiento[j][i]
        xn.append(datos_normalizar(a,b,c,d))
xn = np.array(xn).reshape(np.shape(entrada_prueba))
xn = np.c_[np.ones(p40),xn]
theta = np.array(theta).reshape(np.shape(theta))
p = 0
n = 0
pos = 0
neg = 0
for i in range(p40):
    z = ((theta.T)@xn[i])
    h.append(funcion_sigmoide(z))
    
    print("\nPara el dato de la Iteracion",i+1)
    print("La predicción es:",h[i])
    print("Dato real normalizado:",yn[i])
    print("Dato real sin normalizar:",valores_prueba[i])
    if(valores_prueba[i]==1):
        pos += 1
    else:
        neg += 1
    if(h[i]>0.50):
        print("Valor positivo")
        p += 1
    else:
        print("Valor negativo")
        n += 1

print("\nLa función h(x) está dada por::",theta)
print("\nResultados del modelo:"+"\nValores positivos: {0} | Valores negativos: {1}".format(p,n))
print("\nResultados esperados:"+"\nValores positivos: {0} | Valores negativos: {1}".format(pos,neg))

#Plotting data
plt.scatter(x[:, 0], x[:, 1], marker='o', c=y,s=25, edgecolor='k')
plt.plot(theta)
plt.show()