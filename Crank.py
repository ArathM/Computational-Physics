# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 21:00:14 2021

Tarea 6: Ecuación de Schrödinger dependiente del tiempo. Método Crank-Nicolson. 
 Física Computacional II
@author: Arath Emannuel Marin Ramírez A01651107 Anuar Kafuri Zarzosa A01650826 
"""
# """
# Este código utiliza el método de Crank-Nicolson, un promedio de soluciones
# explícitas e implicitas de Euler por medio de diferencias finitas, para  
# resolver la ecuación de Schrödinger dependiente del tiempo con un perfil
# Gaussiano como función inicial, para el caso del oscilador armónico
# cuántico y el caso del pozo de potencial infinito
# """

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.sparse as sp
import scipy.sparse.linalg as la

N = 1000 #Puntos del grid espacial
M = 1000  #Puntos de la evolución temporal
a = -50  #Límite izquierdo
b = 50    #Límite derecho
dx = (b-a)/N  #Diferencial espacial
dt = 0.01     #Diferencial temporal

#Se define una matriz para guardar los datos de las funciones de onda en todos
#los pasos de tiempo
psiall = np.zeros((M,N),dtype=complex)

#Parámetros de la Gaussiana inicial
x0 = -30
sigma0 = 5
k0 = 3

#Se define una función potencial
def V(x):
    #V= 0
    #V = 1/2*x**2
    
    if -20<x<20:
        V = 5
    else:
        V= 0
    return V

#Se define la función f(x) inicial, en este caso una Gaussiana
x = np.arange(a,b,dx)
psinot = np.array([np.exp(-1/2*((i-x0)/sigma0)**2)*np.exp(1j*k0*i) for i in x])
psi = psinot/np.sqrt(np.trapz(np.absolute(psinot)**2,x))
psi[0] = 0; psi[N-1] = 0

#Formando la matriz tridiagonal
A = np.zeros((N, N),dtype=complex)
A[0,0] = 1 + 1j*dt/(2*dx**2) + 1j*V(a)*dt/2 
A[0,1] = -1j*dt/(4*dx**2)
A[N-1,N-1] = 1 + 1j*dt/(2*dx**2) + 1j*V(b)*dt/2
A[N-1,N-2] = -1j*dt/(4*dx**2)

for i in range(1,N-2):
    A[i,i] = 1 + 1j*dt/(2*dx**2) + 1j*V(a+(i*dx))*dt/2
    A[i,i+1] = -1j*dt/(4*dx**2)
    A[i,i-1] = -1j*dt/(4*dx**2)

fig = plt.figure(1)
ax = plt.axes(xlim=(-50, 50), ylim=(0, 0.4))
l, = ax.plot([], [])

def animate(i):
    #Se guarda A como matriz dispersa para agilizar los cálculos
    Asp = sp.dia_matrix(A) 
    global psi
    Acon = np.matrix.conjugate(A) #Se conjuga A para generar el vector a resolver
    B = np.matmul(Acon,psi)
    
    #Resolviendo el sistema lineal
    psi1 = la.lsqr(Asp, B)[0]
    psiall[i,:] = psi
    psi = psi1
    l.set_data(x,np.absolute(psiall[i,:])**2) #Mandando datos de psi actual
    plt.xlabel('x')
    plt.ylabel(r'$|\psi(x)|^2$')
    plt.title(r'Evolución temporal de $|\psi(x)|^2$')	
    return l,


#Norma de cada punto temporal de psi
norms = [np.trapz(np.absolute(psiall[i,:])**2,x) for i in range(0,len(psiall))]
np.mean(norms)

#Guardando la animación
anim = animation.FuncAnimation(fig, animate, frames=M, interval=10, blit=True)
anim.save('hmmm.gif')
    
