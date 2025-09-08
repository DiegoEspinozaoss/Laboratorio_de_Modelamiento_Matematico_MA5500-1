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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#%%
def graficar_flujos(nodo_i, meses):
    flujos_mensuales = {nodo: [] for nodo in range(0, 2000)}  
    for mes_nombre in meses:
        matriz_mes = cargar_matriz(mes_nombre)
        if matriz_mes is None:
            continue  
        
        for fila in matriz_mes:
            nodo_salida = int(fila[0])  
            nodo_llegada = int(fila[1])  
            particulas = fila[2]  
            
            if nodo_salida == nodo_i:
                flujos_mensuales[nodo_llegada].append(particulas)
    
    plt.figure(figsize=(12, 10))
    for nodo, flujos in flujos_mensuales.items():
        if len(flujos) > 0:  
            while len(flujos) < len(meses):
                flujos.append(0)  
            plt.plot(meses, flujos, marker='o', linestyle='-', label=f"Flujo hacia nodo {nodo}")
    
    plt.title(f"Evolución del flujo de partículas desde el nodo {nodo_i} hacia otros nodos")
    plt.xlabel("Mes")
    plt.ylabel("Número de partículas (flujo)")
    plt.xticks(rotation=45)
    plt.grid(False)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

matriz_enero = cargar_matriz("Enero")
nodos_salida = matriz_enero[:, 0]  
nodos_llegada = matriz_enero[:, 1]  
nodos_unicos = np.unique(np.concatenate([nodos_salida, nodos_llegada]))

print("Número de nodos únicos:", len(nodos_unicos))
indice_nodo = 5  

if indice_nodo < len(nodos_unicos):
    nodo_i = nodos_unicos[indice_nodo]  
    print(f"Graficando flujo para el nodo {nodo_i}")
    graficar_flujos(nodo_i, meses)
else:
    print(f"Índice {indice_nodo} fuera de rango.")






























#%%
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