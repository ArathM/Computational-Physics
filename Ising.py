# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 09:06:24 2021

@author: arath
"""

import matplotlib.pyplot as plt
import numpy as np

#Inicialización de valores necesarios

N = 100 #Tamaño del vector de spins
M = 1000 #Número de iteraciones

B = 1 #0.1      #Magnitud del campo magnético
gub = 0.33      #1.469e-11 #Valor tomado del magneton de bohr y la razón giromagnética del electrón
J = 0.02       #Constante de interacción spin-spin 
kb = 0.1 #0.001 #1.38e-23   #Constante de Boltzmann


#Esta función toma un array de spins (la variable spins) y retorna la 
#energía de ese sistema de acuerdo al modelo de Ising Lenz

def Energy(spins):
    sumspin = 0.
    sumfield = sum(spins)
    
    for i in range(0,N-1):
        sumspin += spins[i]*spins[i+1]
        
    E = -J*sumspin - (gub*B)*sumfield
    
    return E


#Esta función toma como argumento un vector de spins de tamaño N, y retorna el número promedio de dominios
#que hay en un tiempo M
def domaincount(spin):
    
    domains = np.zeros(M)
    domainsizes = np.zeros((N,M))
    
    for i in range(0,M):
        dom = 0            #contador que sirve como índice para ubicarse en el dominio actual
        
        for j in range(0,N-1):
            if spin[j,i] - spin[j+1,i] != 0:   #si no son iguales, significa que cambió de spin, e.g. se forma otro dominio
                dom += 1
            domainsizes[dom,i] += 1
        
        domains[i] = dom  #retorna el número total de dominios en dado instante M
    
    nonz = np.nonzero(domainsizes[:,M-1])      #Índices donde el tamaño de dominios no es cero
    averagedomainsizes = np.average(domainsizes[nonz,M-1])
    
    return averagedomainsizes
        

#Implementación del algoritmo de metrópolis para el modelo de Ising Lenz. 
#Toma como argumento un vector inicial de spins (spin) y retorna su evolución en el tiempo
#En específico, esta función regresa: la matriz de vectores de spin a lo largo de tiempo (plotspin), 
#la energía a lo largo del tiempo (energies), la diferencia entre spins arriba y abajo a lo largo
#del tiempo (numspin) y el número total de spins que cambiaron en la corrida (flipped)
    
def ising_metropolis(spin, T):      
    unflipped = 0    #contador del número de veces que el nuevo valor de spin es rechazado por el algoritmo de metrópolis
    numspin = np.zeros(M)
    Energies = np.zeros(M)
    plotspin = np.zeros((N,M))
    
    #Se itera M número de veces para llevar el sistema a equilibro, aceptando energías menores    
    for i in range(0,M):
        j = np.random.randint(0,N) #Índice aleatorio
        Ea = Energy(spin) #Energía actual
        atr = spin #Se genera un vector de spins para realizar la prueba
        plotspin[:,i] = spin #Se actualiza la matriz con los valores de cada iteración para el plot
        down = np.count_nonzero(spin == -1/2)
        up = np.count_nonzero(spin == 1/2)    
        
        #Valores de prueba, se cambia el spin y se calcula la energía
        atr[j] = -atr[j]    
        Etr = Energy(atr)
        
        
        if Etr <= Ea:
            spin[j] = atr[j]  #Se actualiza el vector original si Etr <= Ea
            
        else:
            R = np.exp(-(Etr-Ea)/(kb*T))
            r = np.random.uniform(0,1)
            
            if R >= r:
                spin[j] = atr[j]
            else:
                spin[j] = spin[j]
                unflipped += 1
                
                
        #Datos para graficar la energía
        Energies[i] = Ea
        numspin[i] = abs(down-up)
        flipped = M - unflipped
    
    
    return plotspin, Energies, numspin, flipped

#Instancias de los métodos para su visualización
Temps = np.linspace(0.1,1500,100)      #Vector con distintas temperaturas a probar
tiempo = np.linspace(1,M+1,M)             #Puntos de tiempo

#Alocación de espacio
plots = np.zeros((N, M, len(Temps)))
flips = np.zeros(len(Temps))
energías = np.zeros((M,len(Temps)))
numspins = np.zeros((M,len(Temps)))
dominios = np.zeros(len(Temps))

#Obteniendo valores para cada una de las temperaturas
for i in range(0,len(Temps)):
    #initial = np.random.choice([-1/2,1/2], size = N) #Vector inicial de spins aleatorios
    initial = np.repeat(1/2,N)                        #Vector inicial de spins 1/2 o -1/2
    ma,mb,mc,md = ising_metropolis(initial, Temps[i])
    plots[:,:,i] = ma
    energías[:,i] = mb
    numspins[:,i] = mc
    flips[i] = md
    dominios[i] = domaincount(ma)
    
    
#Plot del vector de spins como función del tiempo
plt.figure(1)
plt.imshow(plots[:,:,0], cmap = 'Greys', aspect = 'auto', origin='lower')            
plt.xlabel('Tiempo') 
plt.ylabel('Posición de spins') 

#Graficando para cada temperatura
for i in range(0,len(Temps)):
    
    #Energía total del sistema como función del tiempo
    plt.figure(2)
    plt.plot(tiempo, energías[:,i], label ="Temperatura = {:f} K".format(Temps[i]))
    plt.xlabel('Tiempo') 
    plt.ylabel('Energía') 
    plt.legend(loc='best')
    
    #Diferencia entre spins up y down como función del tiempo
    plt.figure(3)
    plt.plot(tiempo, numspins[:,i], label ="Temperatura = {:f} K".format(Temps[i]))
    plt.xlabel('Tiempo') 
    plt.ylabel(r'Diferencia en el número de spins $\frac{1}{2}$ y $-\frac{1}{2}$')     
    plt.legend(loc='best')

#Cambio total de spins como función de la temperatura    
plt.figure(4)
plt.plot(Temps, flips, marker = '.')
plt.xlabel('Temperatura') 
plt.ylabel('Número de veces que una partícula cambió de spin') 
    
#Número promedio de dominios como función de la temperatura
plt.figure(5)
plt.plot(Temps, dominios, marker = '.')
plt.xlabel('Temperatura') 
plt.ylabel('Tamaño promedio de dominios') 


plt.show()
