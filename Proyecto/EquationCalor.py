#!/usr/bin/env python
# coding: utf-8

# In[30]:


'''Las bibliotecas a utilizar, teniendo en cuanta que el time se utiliza para contabilidad el tiempo en la
realización de un cierto código'''
import numpy as np
import matplotlib.pyplot as plt
from time import time
import random


# In[31]:


'''np.zeros llenar mi matriz de ceros inicialmente, el fill rellena la matriz con un valor, en este caso será 
la temperatura inicial llamada (Tini), imprimiendo T puede observar que la matriz T está rellena del valor inicial
excepto las fronteras'''

def condiciones(nx,ny,Ttop,Tbottom,Tleft,Tright,Tini):
    T = np.zeros((nx, ny))
    T.fill(Tini)
#print(T)

# Condiciones de frontera
    T[0, :] = Ttop
    T[-1, :] = Tbottom
    T[:, -1] = Tright
    T[:, 0] = Tleft
    return T


# In[32]:


'''En este caso es de gran utilidad el np.roll ya que en mis valores de frontera tienen el altercado que no poder
realizar diferencias finitas, porque en sus extremos me faltaría un valor para completar dicha operación, 
El np.roll lo soluciona, ese comando desplaza elementos de la matriz como uno lo desee.
Diferencias finitas T[i, j] = T[i,j]+(T[i,j-1]+T[i-1,j]-4*T[i,j]+T[i+1,j]+T[i,j+1])'''

def iteration(T):
    tmp = np.zeros(T.shape)
    tmp=T+0.25*(np.roll(T,1,axis=0)+np.roll(T,-1,axis=0)+np.roll(T,1,axis=1)+np.roll(T,-1,axis=1)-4*T)
    return tmp


# In[33]:


'''Se muestra el proceso de diferencias finitas utilizando for, aunque es claro que el optimo es iteration (con roll)'''
def iteration1(T):
    tmp = np.zeros(T.shape)

    for i in range(1, T.shape[0]-1):
        for j in range(1, T.shape[1]-1):
            tmp[i,j] = T[i,j]+0.25*(T[i,j-1]+T[i-1,j]-4*T[i,j]+T[i+1,j]+T[i,j+1])
    return tmp


# In[34]:


'''Guarda las iteraciones, depende de a iteración que deseeo la puedo llamar con j.
Además, observe que está función es recursiva ya que se llama a sí misma y a partir del valor que retorna
de nuevo lo proceso y va ejecutando la función iteration'''

def guardaT(j,T0):
    P=T0
    for i in range(j):
        if i>=0:
            P=iteration(P)
    return P
    
#Condiciones iniciales
T=condiciones(5,5,0,0,0,0,100)

#Llamando a la función y pidiendo solo la iteración dada
#J=guardaT(4,T)
#print(J)
#plt.imshow(J)


t0=time()
for i in range(3):
    T = iteration(T)
    colorinterpolation=50
    colourMap=plt.cm.jet
    plt.title('Ecuación de calor')
    plt.contourf(T, colorinterpolation, cmap=colourMap)
    plt.colorbar()
    #print(T)
    plt.figure()
plt.imshow(T)
plt.clf()
x=input()
t1=time()
tt=t1-t0
print(tt)
#print(J)


# In[43]:


'''OBJETIVO: Generar tablas con matrices y respectivos gráficos, cada tabla o txt este representada por unas 
condiciones aleatorias aleatorias pero se va a mostrar en todos los tipos de txt la iteración que se desee'''

'''El comando rand.seed() genera números aleatorios, según las fuentes es establecer una semilla'''
    
'''El for i in range que aparece a continuación es la cantidad de tablas que se va a generar, en este caso serán 4 tablas
Cada uno con unas condiciones de frontera diferentes, lo que no va a cambiar es:
La dimensión de la matriz y la condición inicial '''




for i in range(4):
    Te = condiciones(5,5, random.randrange(100,500),random.randrange(10,100),random.randrange(100,500),random.randrange(10,100), 100)

#Es claro que aquí no es necesario llamar la función iteration, ya que a partir de la función guarT esta la llama
#Además esta función es útil porque solo se quiere imprimir una iteración específica.

    Tem= guardaT(3,Te)
    colorinterpolation=50
    colourMap=plt.cm.jet
    plt.title('Ecuación de calor')
    plt.contourf(Tem, colorinterpolation, cmap=colourMap)
    plt.colorbar()
    plt.figure()
    plt.imshow(Tem)
    print(Tem)
    np.savetxt('Tem.txt',Tem,fmt="%s")
    plt.savefig('Temperatura.png')
    


# In[ ]:




