
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter


def cargar_grid():
    path_grid = os.path.join('..', 'Ser Giaconi Matrices', 'networkdomain1_vflow1_depth3_nodesize0250.grid')
    grid = np.loadtxt(path_grid)
    lon, lat = grid[:, 1], grid[:, 2]

    if lat.max() < 2:
        lat = lat * 180
    lat = np.interp(lat, (lat.min(), lat.max()), (30, 45))

    return lon, lat



def cargar_matriz(mes_nombre):
    nombre_matriz = f'15_{mes_nombre}_2002.matrix'
    directorio = os.path.join('..', 'Ser Giaconi Matrices', '2002_monthbymonth')
    path = os.path.join(directorio, nombre_matriz)
    return np.loadtxt(path)

def calcular_FTLE(lon, lat, matriz, tau_dias=15):
    N = len(lon)
    FTLE = np.zeros(N)

    i_idx = matriz[:, 0].astype(int)
    j_idx = matriz[:, 1].astype(int)
    w = matriz[:, 2]

    min_index = min(i_idx.min(), j_idx.min())
    i_idx -= min_index
    j_idx -= min_index

    for i, j, wij in zip(i_idx, j_idx, w):
        if wij <= 0 or i >= N or j >= N:
            continue
        dx = lon[j] - lon[i]
        dy = lat[j] - lat[i]
        dist_ini = 0.05
        dist_fin = np.sqrt(dx**2 + dy**2)
        if dist_fin > 0:
            FTLE[i] += np.log((dist_fin + 1e-6) / dist_ini) / tau_dias

    FTLE = np.nan_to_num(FTLE, nan=0.0)
    FTLE = np.clip(FTLE, 0, 0.3)
    return FTLE


def plot_FTLE(lon, lat, FTLE, mes_nombre):
    plt.figure(figsize=(11, 6))
    cmap = plt.cm.jet
    norm = mcolors.Normalize(vmin=0, vmax=0.15)
    sc = plt.scatter(lon, lat, c=FTLE, cmap=cmap, norm=norm, s=15, edgecolor='none')
    plt.colorbar(sc, label=r'FTLE $\lambda(x_0,t_0,\tau)$ [day$^{-1}$]')
    plt.title(f"Forward FTLE field – {mes_nombre.capitalize()} 2002 (τ = 15 días)")
    plt.xlabel("Longitud"); plt.ylabel("Latitud")
    plt.xlim(-10, 37); plt.ylim(30, 46)
    plt.grid(False)
    plt.show()


mes = "julio"
lon, lat = cargar_grid()

print(f"Min/Max longitud: {lon.min()} – {lon.max()}")
print(f"Min/Max latitud: {lat.min()} – {lat.max()}")

matriz = cargar_matriz(mes)

print(f"Rango de índices i: {int(matriz[:,0].min())} – {int(matriz[:,0].max())}")
print(f"Rango de índices j: {int(matriz[:,1].min())} – {int(matriz[:,1].max())}")
print(f"Número de nodos del grid: {len(lon)}")

FTLE = calcular_FTLE(lon, lat, matriz, tau_dias=15)

FTLE_suave = gaussian_filter(FTLE, sigma=1)

plot_FTLE(lon, lat, FTLE_suave, mes)