# Física Computacional II
# Proyecto final: Gas en el ensamble canónico
# @author: Arath E. Marin Ramírez A01651107 Anuar Kafuri Zarzosa A01650826 
# """
# """ 
# Este codigo realiza dos cosas: Asignar condiciones iniciales de velocidad
# a un gas en el ensamble canónico (T,V,N constantes) según la distribución de
# Maxwell Boltzmann por medio de muestreo de Von Neumann, y de posición 
# según una distribución Gaussiana.
# Después, se realiza la evolución en el tiempo con un potencial 
# intermolecular de Lennard-Jones utilizando el algoritmo de Velocity Verlet, 
# con condiciones de frontera periódicas. 
# """


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

class Canonical:
       
    #Se inicializan las N partículas con dada masa m, el sistema a temperatura 
    #T, a un tiempo total t
    def __init__(self, m, T, N, t):
        self.m = m #6.7e-26
        self.T = T
        self.N = N
        self.t = t
        self.k = 1 #1.38e-23
        self.r = np.zeros((N,3,t))
        self.v = np.zeros((N,3,t))
        self.a = np.zeros((N,3,t))
    
    #Se define la PDF de Maxwell-Boltzmann para asignar velocidades iniciales
    #a las partículas
    def maxwellPDF(self, vel):
        vp = (self.m/(2*np.pi*self.k*self.T)**(1/2))*np.exp(-(self.m*vel**2)/(2*self.k*self.T))
        return vp
    
    #Distribución de rapidez de Maxwell-Boltzmann, se obtiene integrando la
    #distribución de velocidad sobre un ángulo sólido
    def maxwellvelocity(self,vel):
        velo = (self.m/(2*np.pi*self.k*self.T)**(3/2))*(4*np.pi*vel**2)*np.exp(-(self.m*vel**2)/(2*self.k*self.T))
        return velo
        
    #Se asignan velocidades iniciales según la distribución de Maxwell-Boltzmann
    #y posiciones iniciales según una distribución Gaussiana 3D
    def initialize(self):
        
        #Se busca el máximo de la PDF para muestrear por el método de Von Neumann
        self.vs = np.linspace(0,100,1000) #600 para el Argón
        self.PDF = self.maxwellPDF(self.vs) #Se busca el máximo de la PDF
        self.velPDF = self.maxwellvelocity(self.vs)
        Pmax = np.ndarray.max(self.PDF)
        
        #Implementación del muestreo de Von Neumann
        for i in range(0,3):
            for j in range(0,self.N):
                
                accept = 0     
                #Se intenta un valor diferente de P hasta que este entre
                #en la distribución deseada
                
                while accept == 0:
                    ranx = np.random.uniform(0., 100.)
                    Py = np.random.uniform(0, Pmax)                    
                    Px = self.maxwellPDF(ranx)
                     
                    if Py <= Px:
                        self.v[j,i,0] = ranx
                        accept += 1
        
        #Asignando posiciones iniciales según una Gaussiana en tres dimensiones
        #a cada una de las N partículas
        for i in range(0,self.N):
            ranr = np.random.multivariate_normal([0,0,0],(35)*np.identity(3))
            self.r[i,:,0] = ranr
    
    #Se verifica que las velocidades sigan la distribución de Maxwell-Boltzmann
    #tomando K muestras
    def muestreo(self):
        K = 1000
        
        #Se toman K muestras de los componentes del vector velocidad y de
        #la rapidez
        self.sampv = np.zeros((3,K))
        self.samps = np.zeros(K)
        
        for i in range(0,K):
            self.initialize()
            for j in range(0,self.N):
                self.sampv[:,i] = self.v[j,:,0]
                self.samps[i] = np.linalg.norm(self.v[j,:,0])
            
        #Histogramas normalizados graficados con la distribución esperada
        plt.figure(1)
        plt.subplot(221)
        plt.hist(self.sampv[0,:], bins = np.arange(0, 100.0, 1), density = True, label = "Muestreo")
        plt.plot(self.vs,self.PDF/np.trapz(self.PDF,self.vs), label = "Distribución analítica")
        plt.ylabel(r"P($v_x$)")
        plt.xlabel(r"$v_x$")
        plt.legend(prop={'size': 6})

        plt.subplot(222)
        plt.hist(self.sampv[1,:], bins = np.arange(0, 100.0, 1), density = True, label = "Muestreo")
        plt.plot(self.vs,self.PDF/np.trapz(self.PDF,self.vs), label = "Distribución analítica")
        plt.ylabel(r"P($v_y$)")
        plt.xlabel(r"$v_y$")
        plt.legend(prop={'size': 6})
        
        plt.subplot(223)
        plt.hist(self.sampv[2,:], bins = np.arange(0, 100.0, 1), density = True, label = "Muestreo")
        plt.plot(self.vs,self.PDF/np.trapz(self.PDF,self.vs), label = "Distribución analítica")
        plt.ylabel(r"P($v_z$)")
        plt.xlabel(r"$v_z$")
        plt.legend(prop={'size': 6})
        
        plt.subplot(224)
        plt.hist(self.samps, bins = np.arange(0, 100.0, 1), density = True, label = "Muestreo")
        plt.plot(self.vs,self.velPDF/np.trapz(self.velPDF,self.vs), label = "Distribución analítica")
        plt.ylabel(r"$P(|\vec{v}|)$")
        plt.xlabel(r"$|\vec{v}|$")
        plt.legend(prop={'size': 6})
        
        plt.tight_layout()
        
        plt.figure(2)
        plt.plot(self.vs,self.PDF/np.trapz(self.PDF,self.vs), label = "Distribución de velocidad")
        plt.plot(self.vs,self.velPDF/np.trapz(self.velPDF,self.vs), label = "Distribución de rapidez")
        plt.title(r"Distribuciones normalizadas de velocidad (en una dirección) y rapidez")
        plt.ylabel("P(v)")
        plt.xlabel("v")
        plt.legend()
    
    #Distribución inicial 3D de las partículas
    def initr(self):
        fig = plt.figure(3)
        sub1 = fig.add_subplot(111,projection = "3d")
        sub1.view_init(elev=45,azim=45)
        sub1.scatter(self.r[:,0,0],self.r[:,1,0],self.r[:,2,0], s = 100)
        plt.title("Posiciones iniciales de las partículas")
        plt.ylabel("y")
        plt.xlabel("x")
    
    #Se define el vector de fuerza entre partículas según el potencial de
    #Lennard-Jones. Toma dos vectores, su distancia y regresa el vector 
    #fuerza según el potencial mencionado
    def f(self,r1,r2,rn):
        
        if rn<0.01:
            rn = 0.01
            
        lennardf = (48/rn**2)*(1/rn**12-1/(2*rn**6))*(r1-r2)
        return lennardf
    
    #Algoritmo de Velocity-Verlet para la evolución temporal del sistema con 
    #un potencial de Lennard Jones. Además, se aplican condiciones de frontera
    #periódicas para evitar efectos de frontera no deseados. Los parámetros
    #son la longitud en x, y, z del espacio de simulación
    def Verlet(self,lx,ly,lz):
        
        dt = 0.2
        rcutoff = 10
        self.lx = lx
        self.ly = ly
        self.lz = lz
        
        for i in range(0,self.t-1):
            for j in range(0,self.N):
                
                Fsum = np.zeros(3)
                Fsump = np.zeros(3)
                
                for k in range(0,self.N):
                    
                    if k != j:     
                        dis = np.linalg.norm(self.r[j,:,i] - self.r[k,:,i])
                        
                        if dis < rcutoff:
                            Fsum += self.f(self.r[j,:,i],self.r[k,:,i],dis)
                
                self.r[j,:,i+1] = self.r[j,:,i] + dt*self.v[j,:,i] + (dt**2/2)*Fsum
                
                #Condiciones de frontera periódicas
                if self.r[j,0,i+1] < -lx:
                    self.r[j,0,i+1]  += 2*lx
                    
                elif self.r[j,0,i+1] > lx:
                    self.r[j,0,i+1]  -= 2*lx
                    
                if self.r[j,1,i+1] < -ly:
                    self.r[j,1,i+1]  += 2*ly
                    
                elif self.r[j,1,i+1] > ly:
                    self.r[j,1,i+1]  -= 2*ly
                    
                if self.r[j,2,i+1] < -lz:
                    self.r[j,2,i+1]  += 2*lz
                    
                elif self.r[j,2,i+1] > lz:
                    self.r[j,2,i+1]  -= 2*lz
                 
                #Actualizando la fuerza para t+1
                for k in range(0,self.N):
                    
                    if k != j:     
                        dis = np.linalg.norm(self.r[j,:,i+1] - self.r[k,:,i+1])
                        
                        if dis < rcutoff:
                            Fsump += self.f(self.r[j,:,i+1],self.r[k,:,i+1],dis)
                
                self.v[j,:,i+1] = self.v[j,:,i] + dt*(Fsum+Fsump)/2                   
                
    #Animando los datos obtenidos
    def setanim(self,i):
        self.fig = plt.figure(5)
        self.ax = self.fig.add_subplot(111,projection = "3d")
        
        self.ax.set_xlim3d([-self.lx,self.lx])
        self.ax.set_xlabel('X')
        self.ax.set_ylim3d([-self.ly,self.ly])
        self.ax.set_ylabel('Y')  
        self.ax.set_zlim3d([-self.lz,self.lz])
        self.ax.set_zlabel('Z')
        
        self.sct = self.ax.scatter(self.r[:,0,i],self.r[:,1,i],self.r[:,2,i])
        return self.sct, 
    
    def animate(self):
        anim = animation.FuncAnimation(self.fig, self.setanim, frames=self.t, interval=10, blit=True)
        anim.save('Simul.gif')
        
    #Energía del sistema como función del tiempo
    def energy(self):
        self.KE = np.zeros(self.t)
        tv = np.arange(0,self.t,1)
        
        for i in range(0,self.t):
            for j in range(0,self.N):
                self.KE[i] += 1/2*self.m*np.linalg.norm(self.r[j,:,i])
        
        plt.figure(6)
        plt.plot(tv,self.KE)
        plt.ylabel("Energía cinética")
        plt.xlabel("Tiempo")
        
        plt.show()


#Instanciando la clase y los métodos          
s = Canonical(1,172,3,10)
s.initialize()
s.muestreo()
s.initr()
s.Verlet(50,50,50)
s.setanim(0)
s.animate()
s.energy()