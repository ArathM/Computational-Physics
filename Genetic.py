# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 08:51:20 2021

@author: arath
"""

import numpy as np
import matplotlib.pyplot as plt

P = 5 #Número de parametros
M = 50 #Tamaño de la poblacion

#Inicializando la población con dado rango para cada parámetro
population  = np.zeros((M,P))
population[:,4] =  np.random.uniform(-10,10, size = (M))
population[:,3] =  np.random.uniform(-10,20, size = (M))
population[:,2] =  np.random.uniform(-100,100, size = (M))
population[:,1] =  np.random.uniform(-200,200, size = (M))
population[:,0] =  np.random.uniform(-1000,1000, size = (M))

#Se abre el archivo de datos 
data = open("mock_data_ga.txt", "r")
s = data.readlines()

data = np.zeros((100,3))

#Escaneando el archivo de datos y guardando los datos numéricos en una matriz
for i in range(0,len(s)):
    currentline = s[i]
    for j in range(0,3):   
        data[i,j] = float(currentline.split(' ')[j])


#Funcion a ajustar 

def f(a0,a1,a2,a3,a4,x):
    
    f = a0+a1*x+a2*x**2+a3*x**3+a4*x**4 
    
    return f


#Cálculo del valor chi cuadrada de un vector de datos con ciertos parámetros
#del modelo
#Toma como entrada los puntos de variable independiente (R), 
#la variable dependiente (yobs)
#el error de medición (sigma, en este caso es constante) 
#y los parámetros de modelo v0 y Rc
#Retorna el valor de chi cuadrada 
    

def ChiSqr(x, yobs, sigma, a0,a1,a2,a3,a4):
    
    Chisqrsum = 0
    
    for i in range(0,len(x)):
        Chisqr = ((yobs[i] - f(a0,a1,a2,a3,a4,x[i]))/sigma)**2
        Chisqrsum += Chisqr
       
    return 1/Chisqrsum

#Toma como datos los puntos a los cuales se ajustara la curva (datapoints)
#y la poblacion (pop) para regresar una lista con los valores de aptitud para
#cada elemento de la población
    
def aptitud(datapoints, pop):
    
    aptlist = np.zeros(len(pop))
    
    for i in range(0,len(pop)):
        aptlist[i] = ChiSqr(datapoints[:,0], datapoints[:,1], 5, pop[i,0], pop[i,1], pop[i,2], pop[i,3], pop[i,4])
    
    return aptlist
        

#Esta funcion toma la lista de aptitudes, y utilizando el metodo 
#fitness proportional retorna el indice de uno de los padres para usar
    
def parents(aptitudes):
    
    S = sum(aptitudes)            #Suma de las aptitudes
    print(S)
    tosum = np.random.uniform(0,S)  #Se sumará hasta este punto, y se tomará
                                    #como padre el elemento de ese índice
    currentsum = 0             #Suma cumulativa de los valores de las aptitudes
    
    parentindex = 0            #Índice que se retornara para el padre
    
    while currentsum < tosum:
        
        currentsum += aptitudes[parentindex]
        parentindex += 1
    
    return parentindex

#Toma como argumentos la poblacion y la lista con sus respectivas aptitudes 
#para retornar el cruce simple entre dos elementos
    
def cruce(pop, aptitudes):
    
    hijo = np.zeros(P)
    padre = parents(aptitudes)
    madre = parents(aptitudes)
    
    Pa = int(P/2)   #Toma la mitad de los genes del padre, la mitad de la madre
    hijo[:Pa] = pop[padre,:Pa]
    hijo[Pa:] = pop[madre,Pa:]
    
    return hijo
    
def mutacion(gen):
    
    index1 = np.random.randint(0,len(gen))
    index2 = np.random.randint(0,len(gen))
    
    gen[index1] = gen[index2]
    
    return gen

#Esta función implementa el algoritmo genético sobre N iteraciones con los
#métodos definidos arriba para selección de padre, cruce y mutación 
#Cada iteración se lleva a cabo el proceso de escoger 2 padres, cruzarlos,
#mutar al descendiente y reemplazarlo por el elemento de la lista 
#con menor aptitud
    
def algoritmogen(data,population,N):
    
    for i in range(0,N):
        aptitudlist = aptitud(data,population) #lista de aptitudes para 
                                               #cada elemento
        cruces = cruce(population,aptitudlist) #cruce de elementos
        mutacioniter = mutacion(cruces)        #mutación
        
        #Se busca el índice del elemento menos apto y se reemplaza por
        #el nuevo elemento
        menosapto = np.argmin(aptitudlist)   
        population[menosapto,:] = mutacioniter
        
        
algoritmogen(data,population,10)  
    

    
    