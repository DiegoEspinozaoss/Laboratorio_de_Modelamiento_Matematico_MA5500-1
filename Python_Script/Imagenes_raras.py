#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from matplotlib.collections import LineCollection

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

for mes_nombre in meses:
    matriz_mes = cargar_matriz(mes_nombre)
    if matriz_mes is None:
        continue  
    flujo_min = np.min(matriz_mes[:, 2])  
    flujo_max = np.max(matriz_mes[:, 2])  
    cmap = plt.cm.viridis  
    norm = mcolors.Normalize(vmin=0, vmax=50)  
    plt.figure(figsize=(12, 10))
    lines = []
    colors = []

    for fila in matriz_mes:
        nodo_salida = int(fila[0])  
        nodo_llegada = int(fila[1])  
        particulas = fila[2]  
        lat1, lon1 = grid_data[nodo_salida]
        lat2, lon2 = grid_data[nodo_llegada]
        lines.append([(lon1, lat1), (lon2, lat2)])
        colors.append(cmap(norm(particulas)))  

    lc = LineCollection(lines, colors=colors, linewidths=4)  
    ax = plt.gca()
    ax.add_collection(lc)
    plt.scatter(grid_data[:, 1], grid_data[:, 0], c='skyblue', s=.01, marker='s', edgecolor='black', zorder=5)
    plt.colorbar(lc, label='Número de Partículas (Flujo normalizado entre )', ax=ax)
    plt.title(f"Red de Flujo de Partículas para {mes_nombre} (Matriz de Flujo)")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.grid(False)
    plt.show()