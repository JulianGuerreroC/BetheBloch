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


def CB(z, rho, Z, A, eta0, M, I, C, a, m, X1, X0, N): #Curva de Bragg
               
    etai = np.zeros(N)

    beta = np.zeros(N)
        
    dEdxi = np.zeros(N)
        
    E = np.zeros(N)
       
    etai[0] = eta0 #beta gamma inicial
    
    beta[0] = np.sqrt(eta0**2/(eta0**2 + 1)) #beta inicial
    
    E[0] = (eta0 / beta[0] - 1) * M  #energia cinetica inicial
        
    for i in range(0, N-1): #valores del poder de frenado
    
        #dEdxi[i]= dEdx(z, rho, Z, A, eta, M, I, C, a, m, X1, X0, corr)
        
        dEdxi[i] = dEdx( z, rho , Z, A, etai[i], M , I, C, a, m, X1, X0, True)
                
        E[i+1] = E[i] - dEdxi[i]   #perdida de energia por cm
                
        etai[i+1] = E[i+1] / M + 1
        
        if etai[i+1] < 0.1:
            
            break
        
    if etai[i+1] >= 0.1: #añadir el ultimo elemento fuera del bucle
        
        dEdxi[N-1] = dEdx( z, rho , Z, A, etai[N-1], M, I, C, a, m, X1, X0, True)
                
        
    Results = np.stack((E, dEdxi), axis=0) #guardamos en una matriz los resultados
    
    return (Results)

#%%Datos

z = 1 #carga de la particula incidente

Z_Fe = 26 #numero atomico hierro
A_Fe = 56 #numero masico hierro

Z_Cu = 29 #numero atomico  cobre
A_Cu = 63.55 #numero masico cobre

Z_H2O = 7.42 #numero atomico agua
A_H2O = 18.01 #numero masico agua

Ekin = 10000 #MeV, energia cinetica incidente 

rho_Fe = 7.874 #g/cm^3, densidad hierro
rho_Cu = 8.96 #g/cm^3, densidad cobre
rho_H2O = 1 #g/cm^3, densidad agua


#%%Gráficas

N = 10000

eta = np.logspace( -1, 5 , N ) #valores de beta*gamma
beta = np.zeros(N)
gamma = np.zeros(N)

v = np.zeros(N)

#Fe
dEdxi_muon = np.zeros(N) #poder de frenado
dEdxi_muon_nocorr = np.zeros(N)
dEdxi_pion = np.zeros(N)
dEdxi_p = np.zeros(N)
dEdxi_K = np.zeros(N)

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
ax.text(1000, 3, r'$H_2O$', fontsize=20)
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

ax.text(300, 12, 'p', fontsize=12)
ax.text(70, 10, 'K', fontsize=12)
ax.text(50, 8, r'$\pi$', fontsize=12)
ax.text(18, 6, r'$\mu$', fontsize=12)

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

Dx = 1000 #cm, espesor Fe y Cu

N_Bragg = Dx + 1 #numero de puntos

x = np.linspace(0, Dx, N_Bragg) #Profundiad en el material


#Curva de Bragg Fe

#Muon

gamma_muon = Ekin / M_muon + 1
beta_muon = np.sqrt( (gamma_muon**2- 1) / (gamma_muon**2))
eta_muon = beta_muon* gamma_muon

#proton

gamma_p = Ekin / M_p + 1
beta_p = np.sqrt( (gamma_p**2- 1) / (gamma_p**2))
eta_p = beta_p* gamma_p

#Kaon

gamma_K = Ekin / M_K + 1
beta_K = np.sqrt( (gamma_K**2- 1) / (gamma_K**2))
eta_K = beta_K* gamma_K

#Pion

gamma_pion = Ekin / M_pion + 1
beta_pion = np.sqrt( (gamma_pion**2- 1) / (gamma_pion**2))
eta_pion = beta_pion* gamma_pion


#Todo junto Fe

fig, axs = plt.subplots(2)
axs[0].plot( x, CB(1, rho_Fe , Z_Fe, A_Fe, eta_muon, M_muon , 286 * 1e-6, -4.29, 0.1468, 2.96, 3.15, -0.0012, N_Bragg)[0,:] , color = 'blue', label =r'$\mu$')
axs[0].plot( x, CB(1, rho_Fe , Z_Fe, A_Fe, eta_pion, M_pion , 286 * 1e-6, -4.29, 0.1468, 2.96, 3.15, -0.0012, N_Bragg)[0,:] , color = 'violet', label =r'$\pi$')
axs[0].plot( x, CB(1, rho_Fe , Z_Fe, A_Fe, eta_p, M_p , 286 * 1e-6, -4.29, 0.1468, 2.96, 3.15, -0.0012, N_Bragg)[0,:] , color = 'red', label =r'$p$')
axs[0].plot( x, CB(1, rho_Fe , Z_Fe, A_Fe, eta_K, M_K , 286 * 1e-6, -4.29, 0.1468, 2.96, 3.15, -0.0012, N_Bragg)[0,:] , color = 'green', label =r'$K$')
axs[0].text(840, 300, 'p', color = 'black', fontsize = 12)
axs[0].text(750, 100, 'K', color = 'black', fontsize = 12)
axs[0].text(680, 30, r'$\mu$', color = 'black', fontsize = 12)
axs[0].text(730, 40, r'$\pi$', color = 'black', fontsize = 12)
axs[0].set_yscale('log')
axs[0].set_ylabel(r'E (MeV)')
axs[0].grid(True)
axs[0].legend()

axs[1].plot( x, CB(1, rho_Fe , Z_Fe, A_Fe, eta_muon, M_muon , 286 * 1e-6, -4.29, 0.1468, 2.96, 3.15, -0.0012, N_Bragg)[1,:] , color = 'blue')
axs[1].plot( x, CB(1, rho_Fe , Z_Fe, A_Fe, eta_pion, M_pion , 286 * 1e-6, -4.29, 0.1468, 2.96, 3.15, -0.0012, N_Bragg)[1,:] , color = 'violet')
axs[1].plot( x, CB(1, rho_Fe , Z_Fe, A_Fe, eta_p, M_p , 286 * 1e-6, -4.29, 0.1468, 2.96, 3.15, -0.0012, N_Bragg)[1,:] , color = 'red')
axs[1].plot( x, CB(1, rho_Fe , Z_Fe, A_Fe, eta_K, M_K , 286 * 1e-6, -4.29, 0.1468, 2.96, 3.15, -0.0012, N_Bragg)[1,:] , color = 'green')
axs[1].text(860, 120, 'p', color = 'black', fontsize = 12)
axs[1].text(770, 100, 'K', color = 'black', fontsize = 12)
axs[1].text(670, 45, r'$\mu$', color = 'black', fontsize = 12)
axs[1].text(740, 65, r'$\pi$', color = 'black', fontsize = 12)
axs[1].set_ylabel(r'-dE/dx (MeV/cm)')
axs[1].set_xlabel(r'Profundidad en el material (cm)')

axs[0].text(50, 100, r'Fe', fontsize=20)
axs[1].grid(True)
plt.savefig('BraggFe.pdf', bbox_inches='tight')  
    
plt.show()


print("Figura 6: Curva de Bragg Fe")
print("Profundidad de muon = ", x[np.argmax(CB(1, rho_Fe , Z_Fe, A_Fe, eta_muon, M_muon , 286 * 1e-6, -4.29, 0.1468, 2.96, 3.15, -0.0012, N_Bragg)[1,:])], " cm")
print("Profundidad de proton = ", x[np.argmax(CB(1, rho_Fe , Z_Fe, A_Fe, eta_p, M_p , 286 * 1e-6, -4.29, 0.1468, 2.96, 3.15, -0.0012, N_Bragg)[1,:])], " cm")
print("Profundidad de kaon = ", x[np.argmax(CB(1, rho_Fe , Z_Fe, A_Fe, eta_K, M_K , 286 * 1e-6, -4.29, 0.1468, 2.96, 3.15, -0.0012, N_Bragg)[1,:])], " cm")
print("Profundidad de pion = ", x[np.argmax(CB(1, rho_Fe , Z_Fe, A_Fe, eta_pion, M_pion , 286 * 1e-6, -4.29, 0.1468, 2.96, 3.15, -0.0012, N_Bragg)[1,:])], " cm")



#Curva de Bragg Cu


fig, axs = plt.subplots(2)
axs[0].plot( x, CB(1, rho_Cu , Z_Cu, A_Cu, eta_muon, M_muon , 322 * 1e-6, -4.42, 0.1434, 2.90, 3.28, 0.0254, N_Bragg)[0,:] , color = 'blue', label =r'$\mu$')
axs[0].plot( x, CB(1, rho_Cu , Z_Cu, A_Cu, eta_pion, M_pion , 322 * 1e-6, -4.42, 0.1434, 2.90, 3.28, 0.0254, N_Bragg)[0,:] , color = 'violet', label =r'$\pi$')
axs[0].plot( x, CB(1, rho_Cu , Z_Cu, A_Cu, eta_p, M_p , 322 * 1e-6, -4.42, 0.1434, 2.90, 3.28, 0.0254, N_Bragg)[0,:] , color = 'red', label =r'$p$')
axs[0].plot( x, CB(1, rho_Cu , Z_Cu, A_Cu, eta_K, M_K , 322 * 1e-6, -4.42, 0.1434, 2.90, 3.28, 0.0254, N_Bragg)[0,:] , color = 'green', label =r'$K$')
axs[0].text(740, 900, 'p', color = 'black', fontsize = 12)
axs[0].text(690, 90, 'K', color = 'black', fontsize = 12)
axs[0].text(600, 20, r'$\mu$', color = 'black', fontsize = 12)
axs[0].text(660, 40, r'$\pi$', color = 'black', fontsize = 12)
axs[0].set_yscale('log')
axs[0].set_ylabel(r'E (MeV)')
axs[0].grid(True)
axs[0].legend()

axs[1].plot( x, CB(1, rho_Cu , Z_Cu, A_Cu, eta_muon, M_muon , 322 * 1e-6, -4.42, 0.1434, 2.90, 3.28, 0.0254, N_Bragg)[1,:] , color = 'blue')
axs[1].plot( x, CB(1, rho_Cu , Z_Cu, A_Cu, eta_pion, M_pion , 322 * 1e-6, -4.42, 0.1434, 2.90, 3.28, 0.0254, N_Bragg)[1,:] , color = 'violet')
axs[1].plot( x, CB(1, rho_Cu , Z_Cu, A_Cu, eta_p, M_p , 322 * 1e-6, -4.42, 0.1434, 2.90, 3.28, 0.0254, N_Bragg)[1,:] , color = 'red')
axs[1].plot( x, CB(1, rho_Cu , Z_Cu, A_Cu, eta_K, M_K , 322 * 1e-6, -4.42, 0.1434, 2.90, 3.28, 0.0254, N_Bragg)[1,:] , color = 'green')
axs[1].text(780, 120, 'p', color = 'black', fontsize = 12)
axs[1].text(690, 110, 'K', color = 'black', fontsize = 12)
axs[1].text(600, 55, r'$\mu$', color = 'black', fontsize = 12)
axs[1].text(665, 75, r'$\pi$', color = 'black', fontsize = 12)
axs[1].set_ylabel(r'-dE/dx (MeV/cm)')
axs[1].set_xlabel(r'Profundidad en el material (cm)')

axs[0].text(50, 100, r'Cu', fontsize=20)
axs[1].grid(True)
plt.savefig('BraggCu.pdf', bbox_inches='tight')  
    
plt.show()


print("Figura 7: Curva de Bragg Cu")
print("Profundidad de muon = ", x[np.argmax(CB(1, rho_Cu , Z_Cu, A_Cu, eta_muon, M_muon , 322 * 1e-6, -4.42, 0.1434, 2.90, 3.28, 0.0254, N_Bragg)[1,:])], " cm")
print("Profundidad de proton = ", x[np.argmax(CB(1, rho_Cu , Z_Cu, A_Cu, eta_p, M_p , 322 * 1e-6, -4.42, 0.1434, 2.90, 3.28, 0.0254, N_Bragg)[1,:])], " cm")
print("Profundidad de kaon = ", x[np.argmax(CB(1, rho_Cu , Z_Cu, A_Cu, eta_K, M_K , 322 * 1e-6, -4.42, 0.1434, 2.90, 3.28, 0.0254, N_Bragg)[1,:])], " cm")
print("Profundidad de pion = ", x[np.argmax(CB(1, rho_Cu , Z_Cu, A_Cu, eta_pion, M_pion , 322 * 1e-6, -4.42, 0.1434, 2.90, 3.28, 0.0254, N_Bragg)[1,:])], " cm")



#Curva de Bragg H2O

Dx_H2O = 8000 #cm, espesor

N_Bragg_H2O = Dx_H2O + 1 #numero de puntos

x_H2O = np.linspace(0, Dx_H2O, N_Bragg_H2O) #valores de la profundidad


fig, axs = plt.subplots(2)
axs[0].plot( x_H2O, CB(1, rho_H2O , Z_H2O, A_H2O, eta_muon, M_muon , 75 * 1e-6, -3.50, 0.0911, 3.48, 2.80, 0.2400, N_Bragg_H2O)[0,:] , color = 'blue', label =r'$\mu$')
axs[0].plot( x_H2O, CB(1, rho_H2O , Z_H2O, A_H2O, eta_pion, M_pion , 75 * 1e-6, -3.50, 0.0911, 3.48, 2.80, 0.2400, N_Bragg_H2O)[0,:] , color = 'violet', label =r'$\pi$')
axs[0].plot( x_H2O, CB(1, rho_H2O , Z_H2O, A_H2O, eta_p, M_p , 75 * 1e-6, -3.50, 0.0911, 3.48, 2.80, 0.2400, N_Bragg_H2O)[0,:] , color = 'red', label =r'$p$')
axs[0].plot( x_H2O, CB(1, rho_H2O , Z_H2O, A_H2O, eta_K, M_K , 75 * 1e-6, -3.50, 0.0911, 3.48, 2.80, 0.2400, N_Bragg_H2O)[0,:] , color = 'green', label =r'$K$')
axs[0].text(6500, 100, 'p', color = 'black', fontsize = 12)
axs[0].text(6000, 30, 'K', color = 'black', fontsize = 12)
axs[0].text(5400, 6, r'$\mu$', color = 'black', fontsize = 12)
axs[0].text(5900, 10, r'$\pi$', color = 'black', fontsize = 12)
axs[0].set_yscale('log')
axs[0].set_ylabel(r'E (MeV)')
axs[0].grid(True)
axs[0].legend()

axs[1].plot( x_H2O, CB(1, rho_H2O , Z_H2O, A_H2O, eta_muon, M_muon , 75 * 1e-6, -3.50, 0.0911, 3.48, 2.80, 0.2400, N_Bragg_H2O)[1,:] , color = 'blue')
axs[1].plot( x_H2O, CB(1, rho_H2O , Z_H2O, A_H2O, eta_pion, M_pion , 75 * 1e-6, -3.50, 0.0911, 3.48, 2.80, 0.2400, N_Bragg_H2O)[1,:] , color = 'violet')
axs[1].plot( x_H2O, CB(1, rho_H2O , Z_H2O, A_H2O, eta_p, M_p , 75 * 1e-6, -3.50, 0.0911, 3.48, 2.80, 0.2400, N_Bragg_H2O)[1,:] , color = 'red')
axs[1].plot( x_H2O, CB(1, rho_H2O , Z_H2O, A_H2O, eta_K, M_K , 75 * 1e-6, -3.50, 0.0911, 3.48, 2.80, 0.2400, N_Bragg_H2O)[1,:] , color = 'green')
axs[1].text(6700, 20, 'p', color = 'black', fontsize = 12)
axs[1].text(6100, 15, 'K', color = 'black', fontsize = 12)
axs[1].text(5400, 6, r'$\mu$', color = 'black', fontsize = 12)
axs[1].text(5900, 10, r'$\pi$', color = 'black', fontsize = 12)
axs[1].set_ylabel(r'-dE/dx (MeV/cm)')
axs[1].set_xlabel(r'Profundidad en el material (cm)')

axs[0].text(50, 100, r'$H_2O$', fontsize=20)
axs[1].grid(True)
plt.savefig('BraggH2O.pdf', bbox_inches='tight')  
plt.show()

print("Figura 8: Curva de Bragg H2O")
print("Profundidad de muon = ", x_H2O[np.argmax(CB(1, rho_H2O , Z_H2O, A_H2O, eta_muon, M_muon , 75 * 1e-6, -3.50, 0.0911, 3.48, 2.80, 0.2400, N_Bragg_H2O)[1,:])], " cm")
print("Profundidad de proton = ", x_H2O[np.argmax(CB(1, rho_H2O , Z_H2O, A_H2O, eta_p, M_p , 75 * 1e-6, -3.50, 0.0911, 3.48, 2.80, 0.2400, N_Bragg_H2O)[1,:])], " cm")
print("Profundidad de kaon = ", x_H2O[np.argmax(CB(1, rho_H2O , Z_H2O, A_H2O, eta_K, M_K , 75 * 1e-6, -3.50, 0.0911, 3.48, 2.80, 0.2400, N_Bragg_H2O)[1,:])], " cm")
print("Profundidad de pion = ", x_H2O[np.argmax(CB(1, rho_H2O , Z_H2O, A_H2O, eta_pion, M_pion , 75 * 1e-6, -3.50, 0.0911, 3.48, 2.80, 0.2400, N_Bragg_H2O)[1,:])], " cm")



