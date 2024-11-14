"""
Ecuación de Bethe–Bloch. Julian Guerrero Canovas
"""
import numpy as np
import matplotlib.pyplot as plt

#%%Constantes

m_e = 0.511 #MeV/c^2, masa electron

M_muon = 105.66 #MeV/c^2, masa muon

M_pion = 139.57 #MeV/c^2, masa pion

M_p = 938.27 #MeV/c^2, masa proton

M_K = 493.68 #Mev/c^2, masa Kaon

c = 1 #unidades naturales

#e = 1.602e-19 #C, carga electrón, se toma z en unidades de e

#%%Funciones

def dEdx(z, rho, Z, A, eta, M, I, C, a, m, X1, X0, corr): #Bethe-Bloch, poder de frenado
    
    beta = np.sqrt(eta**2/(eta**2 + 1))

    gamma = eta / beta
        
    def delta(C, eta, X0): # correccion de densidad
        
        X = np.log10(eta)
        
        if X < X0:
            
            return 0
            
        if X0 < X < X1:
            
            return 4.6052*X + C + a*(X1-X)**m
        
        if X > X1:
        
            return 4.6052*X + C
    
    def Wmax(M, eta): #transferencia maxima de energia en una sola colision
        
        s = m_e / M
        
        return 2 * m_e * c**2 * eta**2 / (1 + 2*s*np.sqrt(1+eta**2) + s**2)
        
    def Cshell(I, eta): #correccion de capas
        
        if eta >= 0.1: #limite para la correccion
        
            return (0.422377 *eta**(-2) + 0.0304043 *eta**(-4) - 0.00038106*eta**(-6))*1e-6 *I**(2) +(3.850190*eta**(-2) - 0.1667989**eta**(-4) + 0.00157955**eta**(-6))*1e-9 *I**3
        
        else:
            
            print('eta no cumple eta >= 0.1')
            
            return 0
        
    Cte = 0.1535 # MeV cm^2 / g = 2 * np.pi * Na * r_e**2 * m_e * c**2
       
    if corr == True: #con las correcciones
        
        dEdx =  Cte * rho * (z**2 * Z)/(A * beta**2) * (np.log(2 * m_e *beta**2 * gamma**2 * c**2 * Wmax(M,eta) / I**2) - 2 * beta**2 - delta(C,eta,X0) - 2*Cshell(I,eta)/Z)
    
    if corr == False: #sin las correcciones
         
         dEdx =  Cte * rho * (z**2 * Z)/(A * beta**2) * (np.log(2 * m_e *beta**2 * gamma**2 * c**2 * Wmax(M,eta) / I**2) - 2 * beta**2 )
        
    return (dEdx)


def CB(z, rho, Z, A, Ekin, M, I, C, a, m, X1, X0): #Curva de Bragg
                   
    gamma0 = Ekin / M + 1 #gamma incial
    beta0 = np.sqrt( (gamma0**2 - 1) / (gamma0**2)) #beta inicial
    eta0 = beta0 * gamma0       #beta gamma inicial 
    
    eta = eta0 #beta gamma inicial
    
    E0 = (eta0 / beta0 - 1) * M  #energia cinetica inicial
    
    x = 0 #distancia recorrida inicial
    
    i = 0  #indice del bucle
    
    xi = [0] #distancia recorrida inicial
    
    Ei = [E0] #valores de la energia
    
    dEdxi = [] #valores del poder de frenado
        
    while eta > 0.1: #bucle
    
        #dEdxi[i]= dEdx(z, rho, Z, A, eta, M, I, C, a, m, X1, X0, corr)
        
        if eta >= 100: #distincion para precision
            
            step = 0.1 #tamaño del paso
            
            x  = x + step #distancia recorrida en el paso
            
            dEdxx = dEdx( z, rho , Z, A, eta, M , I, C, a, m, X1, X0, True)
                
            E = Ei[i] - step*dEdxx   #perdida de energia  en el paso
                          
        
        if 10 <= eta < 100: #distincion para precision
        
            step = 0.01#tamaño del paso, (ajustar si se necesita)
            
            x = x + step #distancia recorrida en el paso
            
            dEdxx = dEdx( z, rho , Z, A, eta, M , I, C, a, m, X1, X0, True)
                
            E = Ei[i] - step*dEdxx  #perdida de energia en el paso
             
            
        if 1 <= eta < 10: #distincion para precision
        
            step = 0.001 #tamaño del paso, (ajustar si se necesita)
        
            x = x + step #distancia recorrida en el paso
            
            dEdxx = dEdx( z, rho , Z, A, eta, M , I, C, a, m, X1, X0, True)
                
            E = Ei[i] - step*dEdxx  #perdida de energia en el paso
                           
        if 0.1 < eta < 1: #distincion para precision
        
            step = 0.001 #tamaño del paso, (ajustar si se necesita)
            
            x = x + step #distancia recorrida en el paso
            
            dEdxx = dEdx( z, rho , Z, A, eta, M , I, C, a, m, X1, X0, True)
                
            E = Ei[i] - step*dEdxx  #perdida de energia  en el paso
                           
            
        if E <= 0: #terminar el bucle ciando E = 0 o negativa
            
            break
        
        eta = E / M + 1 # actualizar valor beta gamma
        
        xi.append(x) #guardar valor distancia
        
        Ei.append(E) #guardar valor energia
        
        dEdxi.append(dEdxx) #guardar valor dEdx
        
        i = i + 1 #actualizar indice
        
    dEdxi.append(0) #añadir 0 al final para cuadrar longitudes
        
    Results = np.stack((Ei, dEdxi, xi), axis=0) #guardamos en una matriz los resultados
    
    return (Results)

#%%Datos

z = 1 #carga de la particula incidente

Z_Fe = 26 #numero atomico hierro
A_Fe = 56 #numero masico hierro

Z_Cu = 29 #numero atomico  cobre
A_Cu = 63.55 #numero masico cobre

Z_H2O = 7.42 #numero atomico agua
A_H2O = 13 #numero masico agua

rho_Fe = 7.874 #g/cm^3, densidad hierro
rho_Cu = 8.96 #g/cm^3, densidad cobre
rho_H2O = 1 #g/cm^3, densidad agua


#%%Gráficas

N = 10000 #numero de puntos

eta = np.logspace( -1, 5 , N ) #valores de beta*gamma
beta = np.zeros(N) #valores de beta
gamma = np.zeros(N) #valores de gamma

v = np.zeros(N) #valores de la velocidad

#Fe
dEdxi_muon = np.zeros(N) #poder de frenado muon
dEdxi_muon_nocorr = np.zeros(N) #poder de frenado muon sin correccion
dEdxi_pion = np.zeros(N) #poder de frenado pion
dEdxi_p = np.zeros(N) #poder de frenado proton
dEdxi_K = np.zeros(N) #poder de frenado kaon

p_muon = np.zeros(N) #momentos
p_pion = np.zeros(N)
p_p = np.zeros(N)
p_K = np.zeros(N)


#Cu
dEdxi_muon_Cu = np.zeros(N) #poder de frenado
dEdxi_pion_Cu = np.zeros(N)
dEdxi_p_Cu = np.zeros(N)
dEdxi_K_Cu = np.zeros(N)


#H2O
dEdxi_muon_H2O = np.zeros(N) #poder de frenado
dEdxi_pion_H2O = np.zeros(N)
dEdxi_p_H2O = np.zeros(N)
dEdxi_K_H2O = np.zeros(N)


#%%Fe

#Correcciones de Bethe-Bloch solo con muon en hierro

for i in range(0,N):
    
    #dEdxi[i]= dEdx(z, rho, Z, A, eta, M, I, C, a, m, X1, X0, corr)
    
    dEdxi_muon[i] = dEdx( 1, rho_Fe , Z_Fe, A_Fe, eta[i], M_muon , 286 * 1e-6, -4.29, 0.1468, 2.96, 3.15, -0.0012, True)
    dEdxi_muon_nocorr[i] = dEdx( 1, rho_Fe , Z_Fe, A_Fe, eta[i], M_muon , 286 * 1e-6, -4.29, 0.1468, 2.96, 3.15, -0.0012, False)
               
        
#Gráfica Estudio correcciones Bethe-Bloch con muon
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)

ax.plot(eta, dEdxi_muon_nocorr,'blue', linestyle = '--', label = r'sin correc.')
ax.plot(eta, dEdxi_muon,'blue', label = r'con correc.')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel(r'-dE/dx (MeV/cm)')
ax.set_xlabel(r'$\beta \gamma$')

ax.text(10000, 100, r'Fe', fontsize=20)
plt.legend()
plt.savefig('muFe.pdf')

plt.show()

#Coordenadas minimo de ionizacion
print("Figura 1: mu en Fe")
print("Minimo de ionizacion: (eta=", eta[np.argmin(dEdxi_muon)], "; dEdx = ", np.min(dEdxi_muon), ")")


#%%Bethe-Bloch con muon,pion, proton y Kaon
   
for i in range(0,N):
    
    #dEdxi[i]= dEdx(z, rho, Z, A, eta, M, I, C, a, m, X1, X0)
    
    dEdxi_muon[i] = dEdx( 1, rho_Fe , Z_Fe, A_Fe, eta[i], M_muon , 286 * 1e-6, -4.29, 0.1468, 2.96, 3.15, -0.0012, True)
    dEdxi_pion[i] = dEdx( 1, rho_Fe , Z_Fe, A_Fe, eta[i], M_pion , 286 * 1e-6, -4.29, 0.1468, 2.96, 3.15, -0.0012, True)
    dEdxi_p[i] = dEdx( 1, rho_Fe , Z_Fe, A_Fe, eta[i], M_p , 286 * 1e-6, -4.29, 0.1468, 2.96, 3.15, -0.0012, True)
    dEdxi_K[i] = dEdx( 1, rho_Fe , Z_Fe, A_Fe, eta[i], M_K , 286 * 1e-6, -4.29, 0.1468, 2.96, 3.15, -0.0012, True)
    
    beta[i] = np.sqrt(eta[i]**2/(eta[i]**2 + 1))
    
    gamma[i] = eta[i] / beta[i]
    
    v[i] = np.sqrt( (gamma[i]**2-1) /(gamma[i]**2) )
    
    p_muon[i] = gamma[i]* M_muon * v[i]
    p_pion[i] = gamma[i]* M_pion * v[i]
    p_p[i] = gamma[i]* M_p * v[i]
    p_K[i] = gamma[i]* M_K * v[i]
    
    
    
#%%Gráfica Bethe-Bloch
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)

ax.plot(eta, dEdxi_muon,'blue', label = r'$\mu$')
ax.plot(eta, dEdxi_pion,'violet', label = r'$\pi$')
ax.plot(eta, dEdxi_p,'red', label = r'$p$')
ax.plot(eta, dEdxi_K,'green', label = r'$K$')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel(r'-dE/dx (MeV/cm)')
ax.set_xlabel(r'$\beta \gamma$')



# Añadir un nuevo eje que comparte el mismo eje x MUON
newax = fig.add_axes(ax.get_position(), frameon=False)
newax.yaxis.tick_right()
newax.yaxis.set_label_position("right")

# Ocultar las líneas y los bordes del nuevo eje
newax.patch.set_visible(False)
newax.yaxis.set_visible(False)

# Ocultar todos los lados del nuevo eje excepto el inferior
for spinename, spine in newax.spines.items():
    if spinename != 'bottom':
        spine.set_visible(False)

# Ajustar la posición del borde inferior del nuevo eje
newax.spines['bottom'].set_position(('axes', -0.2))

# Graficar en ambos ejes
newax.plot(p_muon, dEdxi_muon, 'blue')
newax.set_yscale('log')
newax.set_xscale('log')
newax.set_xlabel(r'Momento $\mu$ (MeV/c)')



# Añadir un nuevo eje que comparte el mismo eje x PROTON
newax = fig.add_axes(ax.get_position(), frameon=False)
newax.yaxis.tick_right()
newax.yaxis.set_label_position("right")

# Ocultar las líneas y los bordes del nuevo eje
newax.patch.set_visible(False)
newax.yaxis.set_visible(False)

# Ocultar todos los lados del nuevo eje excepto el inferior
for spinename, spine in newax.spines.items():
    if spinename != 'bottom':
        spine.set_visible(False)

# Ajustar la posición del borde inferior del nuevo eje
newax.spines['bottom'].set_position(('axes', -0.4))

# Graficar en ambos ejes
newax.plot(p_p, dEdxi_p, 'red')
newax.set_yscale('log')
newax.set_xscale('log')
newax.set_xlabel(r'Momento p (MeV/c)')



# Añadir un nuevo eje que comparte el mismo eje x KAON
newax = fig.add_axes(ax.get_position(), frameon=False)
newax.yaxis.tick_right()
newax.yaxis.set_label_position("right")

# Ocultar las líneas y los bordes del nuevo eje
newax.patch.set_visible(False)
newax.yaxis.set_visible(False)

# Ocultar todos los lados del nuevo eje excepto el inferior
for spinename, spine in newax.spines.items():
    if spinename != 'bottom':
        spine.set_visible(False)

# Ajustar la posición del borde inferior del nuevo eje
newax.spines['bottom'].set_position(('axes', -0.6))

# Graficar en ambos ejes
newax.plot(p_K, dEdxi_K, 'green')
newax.set_yscale('log')
newax.set_xscale('log')
newax.set_xlabel(r'Momento K (MeV/c)')


# Añadir un nuevo eje que comparte el mismo eje x PION
newax = fig.add_axes(ax.get_position(), frameon=False)
newax.yaxis.tick_right()
newax.yaxis.set_label_position("right")

# Ocultar las líneas y los bordes del nuevo eje
newax.patch.set_visible(False)
newax.yaxis.set_visible(False)

# Ocultar todos los lados del nuevo eje excepto el inferior
for spinename, spine in newax.spines.items():
    if spinename != 'bottom':
        spine.set_visible(False)

# Ajustar la posición del borde inferior del nuevo eje
newax.spines['bottom'].set_position(('axes', -0.8))

# Graficar en ambos ejes
newax.plot(p_pion, dEdxi_pion, 'violet')
newax.set_yscale('log')
newax.set_xscale('log')
newax.set_xlabel(r'Momento $\pi$ (MeV/c)')

ax.text(1000, 200, r'Fe', fontsize=20)
ax.legend()
plt.savefig('partFe.pdf', bbox_inches='tight')

plt.show()


#Coordenadas minimo de ionizacion
print("Figura 2: Particulas en Fe")
print("Minimo de ionizacion: (eta=", eta[np.argmin(dEdxi_muon)], "; dEdx = ", np.min(dEdxi_muon), ")")
print("Momento de muon ionizante = ", p_muon[np.argmin(dEdxi_muon)])
print("Momento de proton ionizante = ", p_p[np.argmin(dEdxi_muon)])
print("Momento de kaon ionizante = ", p_K[np.argmin(dEdxi_muon)])
print("Momento de pion ionizante = ", p_pion[np.argmin(dEdxi_muon)])




#%%Gráfica en funcion de dEdx en funcion del momento

fig, ax = plt.subplots()

ax.plot(p_muon, dEdxi_muon,'blue', label = r'$\mu$')
ax.plot(p_pion, dEdxi_pion,'violet', label = r'$\pi$')
ax.plot(p_p, dEdxi_p,'red', label = r'p')
ax.plot(p_K, dEdxi_K,'green', label = r'K')

ax.text(300, 126, 'p', fontsize=12)
ax.text(70, 100, 'K', fontsize=12)
ax.text(50, 70, r'$\pi$', fontsize=12)
ax.text(18, 55, r'$\mu$', fontsize=12)

ax.text(1000000, 200, r'Fe', fontsize=20)


ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel(r'-dE/dx (MeV/cm)')
ax.set_xlabel(r'Momento (MeV/c)')
ax.legend()
plt.savefig('partFep.pdf')

plt.show()


#%%Fe, Cu y H2O

for i in range(0,N):
    
    #dEdxi[i]= dEdx(z, rho, Z, A, eta, M, I, C, a, m, X1, X0, corr)
    
    dEdxi_muon_Cu[i] = dEdx( 1, rho_Cu , Z_Cu, A_Cu, eta[i], M_muon , 322 * 1e-6, -4.42, 0.1434, 2.90, 3.28, 0.0254, True)
    dEdxi_pion_Cu[i] = dEdx( 1, rho_Cu , Z_Cu, A_Cu, eta[i], M_pion , 322 * 1e-6, -4.42, 0.1434, 2.90, 3.28, 0.0254, True)
    dEdxi_p_Cu[i] = dEdx( 1, rho_Cu , Z_Cu, A_Cu, eta[i], M_p , 322 * 1e-6, -4.42, 0.1434, 2.90, 3.28, 0.0254, True)
    dEdxi_K_Cu[i] = dEdx( 1, rho_Cu , Z_Cu, A_Cu, eta[i], M_K , 322 * 1e-6, -4.42, 0.1434, 2.90, 3.28, 0.0254, True)
   
    dEdxi_muon_H2O[i] = dEdx( 1, rho_H2O , Z_H2O, A_H2O, eta[i], M_muon , 75 * 1e-6, -3.50, 0.0911, 3.48, 2.80, 0.2400, True)
    dEdxi_pion_H2O[i] = dEdx( 1, rho_H2O , Z_H2O, A_H2O, eta[i], M_pion , 75 * 1e-6, -3.50, 0.0911, 3.48, 2.80, 0.2400, True)
    dEdxi_p_H2O[i] = dEdx( 1, rho_H2O , Z_H2O, A_H2O, eta[i], M_p , 75 * 1e-6, -3.50, 0.0911, 3.48, 2.80, 0.2400, True)
    dEdxi_K_H2O[i] = dEdx( 1, rho_H2O , Z_H2O, A_H2O, eta[i], M_K , 75 * 1e-6, -3.50, 0.0911, 3.48, 2.80, 0.2400, True)
    
    beta[i] = np.sqrt(eta[i]**2/(eta[i]**2 + 1))
    gamma[i] = eta[i] / beta[i]
    v[i] = np.sqrt( (gamma[i]**2-1) /(gamma[i]**2) )
    
    p_muon[i] = gamma[i]* M_muon * v[i]
    p_pion[i] = gamma[i]* M_pion * v[i]
    p_p[i] = gamma[i]* M_p * v[i]
    p_K[i] = gamma[i]* M_K * v[i]
    
    
    
    
#%%Gráfica Bethe-Bloch en Fe, Cu y H2O
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)

ax.plot(eta, dEdxi_muon,'blue', label = r'$\mu$')
ax.plot(eta, dEdxi_pion,'violet', label = r'$\pi$')
ax.plot(eta, dEdxi_p,'red', label = r'$p$')
ax.plot(eta, dEdxi_K,'green', label = r'$K$')

ax.plot(eta, dEdxi_muon_Cu,'blue')
ax.plot(eta, dEdxi_pion_Cu,'violet')
ax.plot(eta, dEdxi_p_Cu,'red')
ax.plot(eta, dEdxi_K_Cu,'green')

ax.plot(eta, dEdxi_muon_H2O,'blue')
ax.plot(eta, dEdxi_pion_H2O,'violet')
ax.plot(eta, dEdxi_p_H2O,'red')
ax.plot(eta, dEdxi_K_H2O,'green')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel(r'-dE/dx (MeV/cm)')
ax.set_xlabel(r'$\beta \gamma$')



# Añadir un nuevo eje que comparte el mismo eje x MUON
newax = fig.add_axes(ax.get_position(), frameon=False)
newax.yaxis.tick_right()
newax.yaxis.set_label_position("right")

# Ocultar las líneas y los bordes del nuevo eje
newax.patch.set_visible(False)
newax.yaxis.set_visible(False)

# Ocultar todos los lados del nuevo eje excepto el inferior
for spinename, spine in newax.spines.items():
    if spinename != 'bottom':
        spine.set_visible(False)

# Ajustar la posición del borde inferior del nuevo eje
newax.spines['bottom'].set_position(('axes', -0.2))

# Graficar en ambos ejes
newax.plot(p_muon, dEdxi_muon, alpha=0.0)
newax.set_yscale('log')
newax.set_xscale('log')
newax.set_xlabel(r'Momento $\mu$ (MeV/c)')



# Añadir un nuevo eje que comparte el mismo eje x PROTON
newax = fig.add_axes(ax.get_position(), frameon=False)
newax.yaxis.tick_right()
newax.yaxis.set_label_position("right")

# Ocultar las líneas y los bordes del nuevo eje
newax.patch.set_visible(False)
newax.yaxis.set_visible(False)

# Ocultar todos los lados del nuevo eje excepto el inferior
for spinename, spine in newax.spines.items():
    if spinename != 'bottom':
        spine.set_visible(False)

# Ajustar la posición del borde inferior del nuevo eje
newax.spines['bottom'].set_position(('axes', -0.4))

# Graficar en ambos ejes
newax.plot(p_p, dEdxi_p, 'red', alpha=0.0)
newax.set_yscale('log')
newax.set_xscale('log')
newax.set_xlabel(r'Momento p (MeV/c)')



# Añadir un nuevo eje que comparte el mismo eje x KAON
newax = fig.add_axes(ax.get_position(), frameon=False)
newax.yaxis.tick_right()
newax.yaxis.set_label_position("right")

# Ocultar las líneas y los bordes del nuevo eje
newax.patch.set_visible(False)
newax.yaxis.set_visible(False)

# Ocultar todos los lados del nuevo eje excepto el inferior
for spinename, spine in newax.spines.items():
    if spinename != 'bottom':
        spine.set_visible(False)

# Ajustar la posición del borde inferior del nuevo eje
newax.spines['bottom'].set_position(('axes', -0.6))

# Graficar en ambos ejes
newax.plot(p_K, dEdxi_K, 'green', alpha=0.0)
newax.set_yscale('log')
newax.set_xscale('log')
newax.set_xlabel(r'Momento K (MeV/c)')


# Añadir un nuevo eje que comparte el mismo eje x PION
newax = fig.add_axes(ax.get_position(), frameon=False)
newax.yaxis.tick_right()
newax.yaxis.set_label_position("right")

# Ocultar las líneas y los bordes del nuevo eje
newax.patch.set_visible(False)
newax.yaxis.set_visible(False)

# Ocultar todos los lados del nuevo eje excepto el inferior
for spinename, spine in newax.spines.items():
    if spinename != 'bottom':
        spine.set_visible(False)

# Ajustar la posición del borde inferior del nuevo eje
newax.spines['bottom'].set_position(('axes', -0.8))

# Graficar en ambos ejes
newax.plot(p_pion, dEdxi_pion, 'violet', alpha=0.0)
newax.set_yscale('log')
newax.set_xscale('log')
newax.set_xlabel(r'Momento $\pi$ (MeV/c)')

ax.text(1000, 8, r'Fe', fontsize=20)
ax.text(1000, 40, r'Cu', fontsize=20)
ax.text(1000, 4, r'$H_2O$', fontsize=20)
ax.legend()
plt.savefig('partFeCuH2O.pdf', bbox_inches='tight')

plt.show()


#%%Gráfica en funcion de dEdx en funcion del momento Cu

fig, ax = plt.subplots()

ax.plot(p_muon, dEdxi_muon_Cu,'blue', label = r'$\mu$')
ax.plot(p_pion, dEdxi_pion_Cu,'violet', label = r'$\pi$')
ax.plot(p_p, dEdxi_p_Cu, 'red', label = r'p')
ax.plot(p_K, dEdxi_K_Cu,'green', label = r'K')

ax.text(300, 126, 'p', fontsize=12)
ax.text(70, 100, 'K', fontsize=12)
ax.text(50, 70, r'$\pi$', fontsize=12)
ax.text(18, 55, r'$\mu$', fontsize=12)

ax.text(1000000, 200, r'Cu', fontsize=20)


ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel(r'-dE/dx (MeV/cm)')
ax.set_xlabel(r'Momento (MeV/c)')
ax.legend()
plt.savefig('partCup.pdf')

plt.show()


#Coordenadas minimo de ionizacion
print("Figura 4: Particulas en Cu")
print("Minimo de ionizacion: (eta=", eta[np.argmin(dEdxi_muon_Cu)], "; dEdx = ", np.min(dEdxi_muon_Cu), ")")
print("Momento de muon ionizante = ", p_muon[np.argmin(dEdxi_muon_Cu)])
print("Momento de proton ionizante = ", p_p[np.argmin(dEdxi_muon_Cu)])
print("Momento de kaon ionizante = ", p_K[np.argmin(dEdxi_muon_Cu)])
print("Momento de pion ionizante = ", p_pion[np.argmin(dEdxi_muon_Cu)])


#%%Gráfica en funcion de dEdx en funcion del momento H2O

fig, ax = plt.subplots()

ax.plot(p_muon, dEdxi_muon_H2O,'blue', label = r'$\mu$')
ax.plot(p_pion, dEdxi_pion_H2O,'violet', label = r'$\pi$')
ax.plot(p_p, dEdxi_p_H2O, 'red', label = r'p')
ax.plot(p_K, dEdxi_K_H2O,'green', label = r'K')

ax.text(350, 12, 'p', fontsize=12)
ax.text(85, 10, 'K', fontsize=12)
ax.text(65, 8, r'$\pi$', fontsize=12)
ax.text(25, 6, r'$\mu$', fontsize=12)

ax.text(1000000, 30, r'$H_2O$', fontsize=20)


ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel(r'-dE/dx (MeV/cm)')
ax.set_xlabel(r'Momento (MeV/c)')
ax.legend()
plt.savefig('partH2Op.pdf')

plt.show()


#Coordenadas minimo de ionizacion
print("Figura 5: Particulas en H2O")
print("Minimo de ionizacion: (eta=", eta[np.argmin(dEdxi_muon_H2O)], "; dEdx = ", np.min(dEdxi_muon_H2O), ")")
print("Momento de muon ionizante = ", p_muon[np.argmin(dEdxi_muon_H2O)])
print("Momento de proton ionizante = ", p_p[np.argmin(dEdxi_muon_H2O)])
print("Momento de kaon ionizante = ", p_K[np.argmin(dEdxi_muon_H2O)])
print("Momento de pion ionizante = ", p_pion[np.argmin(dEdxi_muon_H2O)])



#%%Bragg

Ekin = [1000 , 2000, 10000] #MeV, valores de la energia cinetica inicial

Ekin_name = ['1000' , '2000', '10000'] #MeV, valores para usar como nombre

for i in range(0,3): #bucle

    #Curva de Bragg Fe
    
    #Todo junto Fe
    
    muon_Fe = CB(1, rho_Fe , Z_Fe, A_Fe, Ekin[i], M_muon , 286 * 1e-6, -4.29, 0.1468, 2.96, 3.15, -0.0012)
    pion_Fe = CB(1, rho_Fe , Z_Fe, A_Fe, Ekin[i], M_pion , 286 * 1e-6, -4.29, 0.1468, 2.96, 3.15, -0.0012)
    p_Fe = CB(1, rho_Fe , Z_Fe, A_Fe, Ekin[i], M_p , 286 * 1e-6, -4.29, 0.1468, 2.96, 3.15, -0.0012)
    K_Fe = CB(1, rho_Fe , Z_Fe, A_Fe, Ekin[i], M_K , 286 * 1e-6, -4.29, 0.1468, 2.96, 3.15, -0.0012)
    
    
    fig, axs = plt.subplots(2)
    axs[0].plot( muon_Fe[2,:], muon_Fe[0,:] , color = 'blue', label =r'$\mu$')
    axs[0].plot( pion_Fe[2,:], pion_Fe[0,:] , color = 'violet', label =r'$\pi$')
    axs[0].plot( p_Fe[2,:], p_Fe[0,:] , color = 'red', label =r'$p$')
    axs[0].plot( K_Fe[2,:], K_Fe[0,:] , color = 'green', label =r'$K$')
    axs[0].set_yscale('log')
    axs[0].set_ylabel(r'E (MeV)')
    axs[0].grid(True)
    axs[0].legend()
    
    axs[1].plot( muon_Fe[2,:], muon_Fe[1,:] , color = 'blue', label =r'$\mu$')
    axs[1].plot( pion_Fe[2,:], pion_Fe[1,:] , color = 'violet', label =r'$\pi$')
    axs[1].plot( p_Fe[2,:], p_Fe[1,:] , color = 'red', label =r'$p$')
    axs[1].plot( K_Fe[2,:], K_Fe[1,:] , color = 'green', label =r'$K$')
    axs[1].set_ylabel(r'-dE/dx (MeV/cm)')
    axs[1].set_xlabel(r'Profundidad en el material (cm)')
    
    axs[1].text( max(muon_Fe[2,:])/8, max(muon_Fe[1,:])/8, r'Fe', fontsize=20)
    axs[1].grid(True)
    plt.savefig('BraggFe_'+ Ekin_name[i] +'MeV.pdf', bbox_inches='tight')  
        
    plt.show()
    
    print("Figura 6: Curva de Bragg Fe: " + Ekin_name[i] + " MeV")
    print("Profundidad de muon = ", muon_Fe[2,len(muon_Fe[2,:])-1], " cm")
    print("Profundidad de proton = ", p_Fe[2,len(p_Fe[2,:])-1], " cm")
    print("Profundidad de kaon = ", K_Fe[2,len(K_Fe[2,:])-1], " cm")
    print("Profundidad de pion = ", pion_Fe[2,len(pion_Fe[2,:])-1], " cm")
    
    
    #Curva de Bragg Cu
    
    
    muon_Cu = CB(1, rho_Cu , Z_Cu, A_Cu, Ekin[i],  M_muon , 322 * 1e-6, -4.42, 0.1434, 2.90, 3.28, 0.0254)
    pion_Cu = CB(1, rho_Cu , Z_Cu, A_Cu, Ekin[i],  M_pion , 322 * 1e-6, -4.42, 0.1434, 2.90, 3.28, 0.0254)
    p_Cu = CB(1, rho_Cu , Z_Cu, A_Cu, Ekin[i],  M_p , 322 * 1e-6, -4.42, 0.1434, 2.90, 3.28, 0.0254)
    K_Cu = CB(1, rho_Cu , Z_Cu, A_Cu, Ekin[i],  M_K , 322 * 1e-6, -4.42, 0.1434, 2.90, 3.28, 0.0254)
    
    
    fig, axs = plt.subplots(2)
    axs[0].plot( muon_Cu[2,:], muon_Cu[0,:] , color = 'blue', label =r'$\mu$')
    axs[0].plot( pion_Cu[2,:], pion_Cu[0,:] , color = 'violet', label =r'$\pi$')
    axs[0].plot( p_Cu[2,:], p_Cu[0,:] , color = 'red', label =r'$p$')
    axs[0].plot( K_Cu[2,:], K_Cu[0,:] , color = 'green', label =r'$K$')
    axs[0].set_yscale('log')
    axs[0].set_ylabel(r'E (MeV)')
    axs[0].grid(True)
    axs[0].legend()
    
    axs[1].plot( muon_Cu[2,:], muon_Cu[1,:] , color = 'blue', label =r'$\mu$')
    axs[1].plot( pion_Cu[2,:], pion_Cu[1,:] , color = 'violet', label =r'$\pi$')
    axs[1].plot( p_Cu[2,:], p_Cu[1,:] , color = 'red', label =r'$p$')
    axs[1].plot( K_Cu[2,:], K_Cu[1,:] , color = 'green', label =r'$K$')
    axs[1].set_ylabel(r'-dE/dx (MeV/cm)')
    axs[1].set_xlabel(r'Profundidad en el material (cm)')
    
    axs[1].text(max(muon_Cu[2,:])/8, max(muon_Cu[1,:])/8, r'Cu', fontsize=20)
    axs[1].grid(True)
    plt.savefig('BraggCu_'+ Ekin_name[i] +'MeV.pdf', bbox_inches='tight')  
        
    plt.show()
    
    
    print("Figura 7: Curva de Bragg Cu: " + Ekin_name[i] + " MeV")
    print("Profundidad de muon = ", muon_Cu[2,len(muon_Cu[2,:])-1], " cm")
    print("Profundidad de proton = ", p_Cu[2,len(p_Cu[2,:])-1], " cm")
    print("Profundidad de kaon = ", K_Cu[2,len(K_Cu[2,:])-1], " cm")
    print("Profundidad de pion = ", pion_Cu[2,len(pion_Cu[2,:])-1], " cm")
    
    
    #Curva de Bragg H2O
    
    muon_H2O = CB(z, rho_H2O , Z_H2O, A_H2O, Ekin[i], M_muon , 75 * 1e-6, -3.50, 0.0911, 3.48, 2.80, 0.2400)
    pion_H2O = CB(z, rho_H2O , Z_H2O, A_H2O, Ekin[i], M_pion , 75 * 1e-6, -3.50, 0.0911, 3.48, 2.80, 0.2400)
    p_H2O = CB(z, rho_H2O , Z_H2O, A_H2O, Ekin[i], M_p , 75 * 1e-6, -3.50, 0.0911, 3.48, 2.80, 0.2400)
    K_H2O = CB(z, rho_H2O , Z_H2O, A_H2O, Ekin[i], M_K , 75 * 1e-6, -3.50, 0.0911, 3.48, 2.80, 0.2400)
    
    fig, axs = plt.subplots(2)
    axs[0].plot( muon_H2O[2,:], muon_H2O[0,:] , color = 'blue', label =r'$\mu$')
    axs[0].plot( pion_H2O[2,:], pion_H2O[0,:] , color = 'violet', label =r'$\pi$')
    axs[0].plot( p_H2O[2,:], p_H2O[0,:] , color = 'red', label =r'$p$')
    axs[0].plot( K_H2O[2,:], K_H2O[0,:] , color = 'green', label =r'$K$')
    axs[0].set_yscale('log')
    axs[0].set_ylabel(r'E (MeV)')
    axs[0].grid(True)
    axs[0].legend()
    
    axs[1].plot( muon_H2O[2,:], muon_H2O[1,:] , color = 'blue', label =r'$\mu$')
    axs[1].plot( pion_H2O[2,:], pion_H2O[1,:] , color = 'violet', label =r'$\pi$')
    axs[1].plot( p_H2O[2,:], p_H2O[1,:] , color = 'red', label =r'$p$')
    axs[1].plot( K_H2O[2,:], K_H2O[1,:] , color = 'green', label =r'$K$')
    axs[1].set_ylabel(r'-dE/dx (MeV/cm)')
    axs[1].set_xlabel(r'Profundidad en el material (cm)')
    
    axs[1].text(max(muon_H2O[2,:])/8, max(muon_H2O[1,:])/8, r'$H_2O$', fontsize=20)
    axs[1].grid(True)
    plt.savefig('BraggH2O_'+ Ekin_name[i] +'MeV.pdf', bbox_inches='tight')  
    plt.show()
    
    print("Figura 8: Curva de Bragg H2O: " + Ekin_name[i] + " MeV")
    print("Profundidad de muon = ", muon_H2O[2,len(muon_H2O[2,:])-1], " cm")
    print("Profundidad de proton = ", p_H2O[2,len(p_H2O[2,:])-1], " cm")
    print("Profundidad de kaon = ", K_H2O[2,len(K_H2O[2,:])-1], " cm")
    print("Profundidad de pion = ", pion_H2O[2,len(pion_H2O[2,:])-1], " cm")


#PSTAR Data, utiles para comprobar resultados
#Protons in Fe
#6.382e3 /rho_Fe #10000 MeV
#1.127e3 /rho_Fe #2000 MeV
#3.254e2 /rho_Fe #1000 MeV

#Protons in Cu
#6.600e3 /rho_Cu #10000 MeV
#1.169e3 /rho_Cu #2000 MeV
#4.757e2 /rho_Cu #1000 MeV

#Protons in H2O
#4.700e3 /rho_H2O #10000 MeV
#8.054e2 /rho_H2O #2000 MeV
#7.889e2 /rho_H2O #1000 MeV

