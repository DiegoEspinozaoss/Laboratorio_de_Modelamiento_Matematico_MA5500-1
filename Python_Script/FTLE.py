#%%

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.sparse import csr_matrix 
from scipy.ndimage import gaussian_filter

def cargar_matriz(mes_nombre):
    nombre_matriz = f'15_{mes_nombre}_2002.matrix'
    directorio_matrices = os.path.join('..', 'Ser Giaconi Matrices', '2002_monthbymonth')
    path = os.path.join(directorio_matrices, nombre_matriz)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encuentra {path}")
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data

def cargar_grid(path_grid=None):
    """
    Lee grid con estructura: lat lon (ya en grados), ignora columnas extra.
    Devuelve lon, lat del mismo tamaño (~8000 puntos).
    """
    import numpy as np, os

    if path_grid is None:
        path_grid = os.path.join('..', 'Ser Giaconi Matrices', 'networkdomain1_vflow1_depth3_nodesize0250.grid')

    data = np.loadtxt(path_grid)

    lat = data[:, 0].astype(float)
    lon = data[:, 1].astype(float)

    if np.any(lat < -90) or np.any(lat > 90):
        lat = (lat - np.min(lat)) / (np.max(lat) - np.min(lat)) * 180 - 90
    if np.any(lon < -180) or np.any(lon > 180):
        lon = (lon - np.min(lon)) / (np.max(lon) - np.min(lon)) * 360 - 180

    print(f"Grid cargado correctamente: {len(lon)} puntos")
    print(f"Latitud: {lat.min():.2f} – {lat.max():.2f}")
    print(f"Longitud: {lon.min():.2f} – {lon.max():.2f}")
    return lon, lat


def normaliza_indices_matriz(i, j, lon, lat):
    """
    Detecta si i/j están en 1-based y los pasa a 0-based,
    y recorta lon/lat para que tengan al menos max_index+1 elementos válidos.
    """
    i = i.astype(int).ravel()
    j = j.astype(int).ravel()

    one_based = False
    if i.min() == 1 or j.min() == 1:
        one_based = True
    if one_based:
        i = i - 1
        j = j - 1

    max_idx = int(max(i.max(), j.max()))
    if len(lon) <= max_idx or len(lat) <= max_idx:
        raise ValueError(f"Grid insuficiente: grid tiene {len(lon)} puntos y se requiere índice {max_idx}.")

    return i, j

def recorta_para_plot(lon, lat, values):
    """
    Recorta/filtra lon, lat y values para:
    - Igualar longitudes
    - Remover NaN
    - Evitar errores de Matplotlib
    """
    n = min(len(lon), len(lat), len(values))
    lon = lon[:n]; lat = lat[:n]; values = values[:n]
    mask = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(values)
    lon = lon[mask]; lat = lat[mask]; values = values[mask]
    return lon, lat, values

def calcular_FTLE(lon, lat, matriz, tau_dias=15):
    """
    Proxy de FTLE basado en separación entre nodos conectados ponderada por el flujo.
    No es el FTLE continuo del modelo NEMO, pero sirve para correlación relativa.
    """
    N = min(len(lon), len(lat))
    i_idx = matriz[:, 0].astype(int)
    j_idx = matriz[:, 1].astype(int)
    w = matriz[:, 2].astype(float)

    if i_idx.min() == 1 or j_idx.min() == 1:
        i_idx -= 1
        j_idx -= 1

    FTLE = np.zeros(N, dtype=float)
    for ii, jj, wij in zip(i_idx, j_idx, w):
        if wij <= 0 or ii >= N or jj >= N:
            continue
        dx = lon[jj] - lon[ii]
        dy = lat[jj] - lat[ii]
        dist_fin = np.hypot(dx, dy)
        dist_ini = 0.05
        if dist_fin > 0:
            FTLE[ii] += np.log((dist_fin + 1e-9) / dist_ini) / tau_dias

    FTLE = np.nan_to_num(FTLE, nan=0.0, posinf=0.0, neginf=0.0)
    FTLE = np.clip(FTLE, 0, np.percentile(FTLE, 99))  
    return FTLE

def plot_scatter_map(lon, lat, values, title, cbar_label, vmin=None, vmax=None, fname=None):
    lon_p, lat_p, val_p = recorta_para_plot(lon, lat, values)
    plt.figure(figsize=(11, 6))
    cmap = plt.cm.jet
    if vmin is None: vmin = np.nanmin(val_p)
    if vmax is None: vmax = np.nanmax(val_p)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sc = plt.scatter(lon_p, lat_p, c=val_p, cmap=cmap, norm=norm, s=12, edgecolor='none')
    plt.colorbar(sc, label=cbar_label)
    plt.title(title)
    plt.xlabel("Longitud"); plt.ylabel("Latitud")
    plt.xlim(-10, 37); plt.ylim(30, 46)
    plt.grid(False)
    if fname:
        plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.show()

mes = "julio"
tau_dias = 15


matriz = cargar_matriz(mes)                  
lon_all, lat_all = cargar_grid()
idx_map = None  
  


i = matriz[:, 0].astype(int)
j = matriz[:, 1].astype(int)
w = matriz[:, 2].astype(float)


i, j = normaliza_indices_matriz(i, j, lon_all, lat_all)
N = int(max(i.max(), j.max()) + 1)


lon = lon_all[:N].copy()
lat = lat_all[:N].copy()


A = csr_matrix((w, (i, j)), shape=(N, N))
rowsum = np.asarray(A.sum(axis=1)).ravel()
rowsum[rowsum == 0] = 1.0
P = A.multiply((1.0 / rowsum)[:, None]).tocsr()



P_dense = P.toarray()
H1 = np.zeros(N, dtype=float)
for r in range(N):
    p = P_dense[r]
    m = p > 0
    if m.any():
        H1[r] = -np.sum(p[m] * np.log(p[m])) / tau_dias
H1 = np.nan_to_num(H1, nan=0.0)


plot_scatter_map(
    lon, lat, H1,
    title=f"Fig. 6 – Entropía de red $H^1_i$ – {mes.capitalize()} 2002 (τ={tau_dias} días)",
    cbar_label=r'$H^1_i(t_0,s)$ [day$^{-1}$]',
    vmin=np.percentile(H1, 1),
    vmax=np.percentile(H1, 99),
    fname=f"Fig6_H1_{mes}.png"
)


FTLE = calcular_FTLE(lon, lat, matriz, tau_dias=tau_dias)
FTLE_suave = gaussian_filter(FTLE, sigma=1)

plot_scatter_map(
    lon, lat, FTLE_suave,
    title=f"Fig. 7 – Proxy FTLE promedio – {mes.capitalize()} 2002 (τ={tau_dias} días)",
    cbar_label=r'$\bar{k}_i(t_0,s)$ (proxy FTLE) [day$^{-1}$]',
    vmin=np.percentile(FTLE_suave, 1),
    vmax=np.percentile(FTLE_suave, 99),
    fname=f"Fig7_FTLE_{mes}.png"
)



n = min(len(H1), len(FTLE_suave))
x = FTLE_suave[:n]
y = H1[:n]
mask = np.isfinite(x) & np.isfinite(y)
x = x[mask]; y = y[mask]

plt.figure(figsize=(6.2, 5.2))
plt.scatter(x, y, s=8, alpha=0.5)
plt.xlabel(r'$\bar{k}_i(t_0,s)$ (proxy FTLE)')
plt.ylabel(r'$H^1_i(t_0,s)$')
plt.title("Fig. 8 – Correlación H1 vs Proxy FTLE (forward)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"Fig8_corr_forward_{mes}.png", dpi=200)
plt.show()


P_back = P.T.tocsr().copy()
colsum = np.asarray(P_back.sum(axis=1)).ravel()
colsum[colsum == 0] = 1.0
P_back = P_back.multiply((1.0 / colsum)[:, None]).tocsr()

P_back_dense = P_back.toarray()
H1_back = np.zeros(N, dtype=float)
for r in range(N):
    p = P_back_dense[r]
    m = p > 0
    if m.any():
        H1_back[r] = -np.sum(p[m] * np.log(p[m])) / tau_dias
H1_back = np.nan_to_num(H1_back, nan=0.0)


n = min(len(H1_back), len(FTLE_suave))
xb = FTLE_suave[:n]
yb = H1_back[:n]
maskb = np.isfinite(xb) & np.isfinite(yb)
xb = xb[maskb]; yb = yb[maskb]

plt.figure(figsize=(6.2, 5.2))
plt.scatter(xb, yb, s=8, alpha=0.5, color='orange')
plt.xlabel(r'$\bar{k}_i(t_0+s,-s)$ (proxy FTLE backward)')
plt.ylabel(r'$H^1_i(t_0+s,-s)$')
plt.title("Fig. 9 – Correlación H1 vs Proxy FTLE (backward)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"Fig9_corr_backward_{mes}.png", dpi=200)
plt.show()

print("Figuras guardadas: Fig6_H1_, Fig7_FTLE_, Fig8_corr_forward_, Fig9_corr_backward_")








path_grid = os.path.join('..', 'Ser Giaconi Matrices', 'networkdomain1_vflow1_depth3_nodesize0250.grid')

with open(path_grid, "r", encoding="utf-8", errors="ignore") as f:
    for k in range(10):
        print(f"{k+1:02d}:", f.readline().rstrip())
