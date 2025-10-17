#%%
import numpy as np
import matplotlib.pyplot as plt
import os

def cargar_matriz(mes_nombre):
    nombre_matriz = f'15_{mes_nombre}_2002.matrix'
    directorio_matrices = os.path.join('..', 'Ser Giaconi Matrices', '2002_monthbymonth')
    path = os.path.join(directorio_matrices, nombre_matriz)
    if os.path.exists(path):
        matriz = np.loadtxt(path)
        return matriz
    else:
        print(f"Archivo no encontrado: {nombre_matriz}")
        return None


def cargar_grid():
    path_grid = os.path.join('..', 'Ser Giaconi Matrices', 'networkdomain1_vflow1_depth3_nodesize0250.grid')
    grid_data = np.loadtxt(path_grid)
    latitudes_longitudes = grid_data[:, :2]  
    return latitudes_longitudes

grid_data = cargar_grid()
meses = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", 
         "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]


def graficar_distribucion_particulas(mes_nombre):
    matriz_mes = cargar_matriz(mes_nombre)    
    if matriz_mes is None:
        return  
    particulas = matriz_mes[:, 2]
    particulas = particulas[particulas > 5]
    plt.figure(figsize=(10, 6))
    plt.hist(particulas, bins=300, color='skyblue', edgecolor='black')
    plt.title(f"Distribución de partículas en los flujos para el mes de {mes_nombre}")
    plt.xlabel("Número de partículas")
    plt.ylabel("Frecuencia")
    plt.yscale('log')  
    plt.show()

for mes in meses:
    graficar_distribucion_particulas(mes)