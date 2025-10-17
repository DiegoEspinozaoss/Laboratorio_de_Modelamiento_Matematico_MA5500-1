#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
    lon = grid_data[:, 1]
    lat = grid_data[:, 2]

    if lon.max() < 10:  
        lon = lon * 360 - 180
    if lat.max() < 2:   
        lat = (lat * 180) - 90
    return lon, lat


def calcular_grados(matriz, N=None):
    i, j, w = matriz[:, 0].astype(int), matriz[:, 1].astype(int), matriz[:, 2]
    N = int(N or max(i.max(), j.max()) + 1)
    out_degree = np.zeros(N)
    in_degree = np.zeros(N)
    for k in range(len(w)):
        out_degree[i[k]] += 1
        in_degree[j[k]] += 1
    return in_degree, out_degree
def plot_degree_map(lon, lat, degree, title):
    n = min(len(lon), len(degree))  
    lon, lat, degree = lon[:n], lat[:n], degree[:n]

    plt.figure(figsize=(9, 6))
    cmap = plt.cm.turbo
    norm = mcolors.Normalize(vmin=0, vmax=40)
    sc = plt.scatter(lon, lat, c=degree, cmap=cmap, s=25, edgecolor='none')
    plt.colorbar(sc, label='Degree (número de conexiones)')
    plt.title(title)
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.grid(False)
    plt.show()

mes = "julio"  
matriz = cargar_matriz(mes)
lon, lat = cargar_grid()

if matriz is not None:
    in_deg, out_deg = calcular_grados(matriz)
    plot_degree_map(lon, lat, in_deg, f"In-degree $K_I(i)$ – {mes.capitalize()} 2002")
    plot_degree_map(lon, lat, out_deg, f"Out-degree $K_O(i)$ – {mes.capitalize()} 2002")
    