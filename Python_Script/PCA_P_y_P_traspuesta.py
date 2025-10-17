#%%
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import random
 
SEED = 42 

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["OMP_NUM_THREADS"] = "1"

random.seed(SEED)
np.random.seed(SEED)



def load_transport_matrix(edge_path, N=None, one_indexed=True):
    """
    Lee una matriz de transporte desde archivo con líneas: 'i j count'
    Devuelve P (CSR) estocástica por filas: P = D^{-1} A
    """
    data = np.loadtxt(edge_path, dtype=float)
    if data.ndim == 1:  
        data = data.reshape(1, -1)
    i, j, w = data[:,0].astype(int), data[:,1].astype(int), data[:,2]
    if one_indexed:
        i -= 1; j -= 1
    N = int(N or max(i.max(), j.max()) + 1)
    A = csr_matrix((w, (i, j)), shape=(N, N))
    rowsum = np.asarray(A.sum(axis=1)).ravel()
    rowsum[rowsum == 0] = 1.0  
    P = A.multiply((1.0 / rowsum)[:, None]).tocsr()
    return P


def plot_eigs_complex(P, k=200, title="Espectro de autovalores en el plano complejo"):
    """
    Calcula y grafica los k autovalores dominantes (de mayor magnitud) de la matriz de transporte P.
    Importante: no se calculan los N autovalores (N puede ser miles),
    solo los más dominantes, ya que calcular todos consumiría mucha memoria y tiempo.
    
    Interpretación:
    - El autovalor λ=1 representa la componente estacionaria (estado de equilibrio).
    - Los autovalores con |λ| < 1 representan modos de relajación:
      indican la velocidad con la que el sistema converge al equilibrio.
      Cuanto más cercano a 1 esté |λ|, más lento es ese modo.
    """
    N = P.shape[0]
    k = min(k, max(1, N-2))
    print(f"\nCalculando los {k} autovalores dominantes de un total de {N} posibles...")
    print("  (Los autovalores dominantes capturan la dinámica global; calcular todos sería costoso en tiempo y memoria.)\n")

    vals = eigs(P, k=k, which='LM', return_eigenvectors=False)

    plt.figure(figsize=(6,6))
    plt.scatter(vals.real, vals.imag, s=16)

    t = np.linspace(0, 2*np.pi, 400)
    plt.plot(np.cos(t), np.sin(t), linewidth=1)
    plt.axhline(0, linewidth=0.5)
    plt.axvline(0, linewidth=0.5)
    plt.gca().set_aspect('equal', 'box')
    plt.title(f"{title}\n({k} autovalores dominantes de {N} posibles)")
    plt.xlabel('Re(λ)')
    plt.ylabel('Im(λ)')
    plt.grid(True, alpha=0.3)
    plt.show()

    print("""
    Interpretación espectral:
    - λ = 1  → componente estacionaria: el sistema está en equilibrio (flujo neto cero).
    - 0 < |λ| < 1 → modos de relajación: describen cómo se disipan o mezclan las trayectorias.
    - |λ| cercano a 1 → relajación lenta (estructuras persistentes).
    - |λ| pequeño → relajación rápida (zonas bien mezcladas).
    """)

    return vals



def spectral_clustering_from_P(P, n_vecs=6, n_clusters=6, random_state=SEED):
    """
    Usa los n_vecs autovectores dominantes (excluyendo el trivial) para embebido espectral y KMeans.
    Devuelve labels (nodos -> cluster), emb (matriz de features para cada nodo).
    """
    N = P.shape[0]
    n_vecs = max(2, n_vecs)
    
    k = min(n_vecs + 1, N-2)
    vals, vecs = eigs(P, k=k, which='LM')
    
    idx = np.argsort(np.abs(vals))[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    
    trivial_idx = np.argmin(np.abs(vals - 1))
    mask = np.ones(len(vals), dtype=bool)
    mask[trivial_idx] = False
    vals_nt = vals[mask][:n_vecs]
    vecs_nt = vecs[:, mask][:, :n_vecs]

    feat = np.hstack([vecs_nt.real, vecs_nt.imag])  

    norms = np.linalg.norm(feat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    feat = feat / norms

    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=SEED)
    labels = km.fit_predict(feat)
    return labels, feat, vals, vecs

def plot_clusters_pca(feat, labels, title="Clustering espectral (PCA de autovectores)", 
                      vals=None, vecs=None):
    """
    Visualiza el embedding espectral reducido a 2D con PCA.
    Además imprime todas las dimensiones y explicaciones matemáticas:
    - número de nodos
    - número de autovalores/autovectores
    - dimensiones del embedding espectral
    - dimensiones reducidas por PCA
    """

    print("\n================= INFORME DE ESTRUCTURA MATEMÁTICA =================\n")
    
    if vals is not None and vecs is not None:
        print(" MATRIZ ESPECTRAL (valores propios y vectores propios)")
        print(f"- Cantidad de autovalores calculados: {len(vals)}")
        print(f"- Cantidad de autovectores calculados: {vecs.shape[1]}")
        print(f"- Dimensión de cada autovector: {vecs.shape[0]}")

        total_theoretical = vecs.shape[0]
        computed = vecs.shape[1]
        percent_used = (computed / total_theoretical) * 100

        print(f"- Cantidad total posible de autovectores (tamaño de P): {total_theoretical}")
        print(f"- Autovectores no calculados: {total_theoretical - computed}")
        print(f"- Porcentaje de autovectores usados: {percent_used:.4f}%")
        print("  (Solo se calcularon los modos dominantes, que capturan la dinámica global del sistema.)")
        print("  (El resto de autovectores representan modos de variación menores o ruido numérico.)")
        print(f"  (Cada autovector tiene un valor por nodo del sistema, total {vecs.shape[0]} nodos.)")
    else:
        print("No se pasaron 'vals' ni 'vecs'. Solo se mostrará información del embedding PCA.")

    N, D = feat.shape
    print("\nEMBEDDING ESPECTRAL")
    print(f"- Número de nodos (filas): {N}")
    print(f"- Dimensiones originales (columnas): {D}")
    print(f"- Estructura de features: {D//2} autovectores → partes real e imaginaria concatenadas")
    print("  (Cada nodo se representa en un espacio de 2×n_vecs dimensiones)")
    print("  (Esto preserva información de fase y magnitud de los modos complejos del sistema)")
    
    pca = PCA(n_components=2, random_state=SEED)
    xy = pca.fit_transform(feat)
    var_exp = pca.explained_variance_ratio_

    
    print("\nPCA (Análisis de Componentes Principales)")
    print(f"- Componentes usados: 2")
    print(f"- Varianza explicada por PC1: {var_exp[0]*100:.2f}%")
    print(f"- Varianza explicada por PC2: {var_exp[1]*100:.2f}%")
    print(f"- Varianza total capturada: {(var_exp[0]+var_exp[1])*100:.2f}%")
    print(f"- Varianza no capturada (resto de dimensiones): {(1 - var_exp[:2].sum())*100:.2f}%")
    print("\n  PC1 y PC2 son combinaciones lineales ortogonales de las dimensiones originales.")
    print("  Cada eje del gráfico corresponde a una dirección de máxima varianza en el embedding.")
    print("  En este plano, los puntos más separados reflejan nodos con dinámicas espectrales distintas.\n")

    print("=====================================================================\n")
    
    plt.figure(figsize=(6.8,5.4))
    for c in np.unique(labels):
        m = labels == c
        plt.scatter(xy[m,0], xy[m,1], s=8, label=f"cluster {c}")

    plt.title(f"{title}\nVarianza explicada: PC1 = {var_exp[0]*100:.2f}% | PC2 = {var_exp[1]*100:.2f}%")
    plt.xlabel(f"PC1 ({var_exp[0]*100:.2f}% varianza)")
    plt.ylabel(f"PC2 ({var_exp[1]*100:.2f}% varianza)")
    plt.legend(markerscale=2, fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.show()



def load_grid_coords(grid_path, one_indexed=True):
    """
    Lee archivo .grid con mapping: idx lon lat ... (los dos últimos números pueden ignorarse).
    Devuelve arrays lon, lat con orden por índice.
    """
    raw = []
    with open(grid_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts: 
                continue
            if len(parts) < 3:
                continue
            try:
                idx = int(parts[0])
                lon = float(parts[1])
                lat = float(parts[2])
                raw.append((idx, lon, lat))
            except:
                continue
    if not raw:
        raise ValueError("No se pudieron leer coordenadas desde el .grid")

    raw = np.array(raw, dtype=float)
    idx = raw[:,0].astype(int)
    if one_indexed:
        idx -= 1
    N = idx.max() + 1
    lon = np.zeros(N); lat = np.zeros(N)
    lon[idx] = raw[:,1]; lat[idx] = raw[:,2]
    return lon, lat

def plot_clusters_geo(lon, lat, labels, title="Clusters sobre mapa (lon/lat)"):
    """
    Dispersa nodos por lon/lat coloreados por cluster. (Mapa simple sin cartopy.)
    """
    plt.figure(figsize=(7,5.5))
    for c in np.unique(labels):
        m = labels == c
        plt.scatter(lon[m], lat[m], s=6, label=f"cluster {c}")
    plt.title(title)
    plt.xlabel("Longitud"); plt.ylabel("Latitud")
    plt.legend(markerscale=2, fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.show()

meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
         "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]

if __name__ == "__main__":
    mes_nombre = meses[0] 
    nombre_matriz = f"15_{mes_nombre}_2002.matrix"
    directorio_matrices = os.path.join("..", "Ser Giaconi Matrices", "2002_monthbymonth")
    path = os.path.join(directorio_matrices, nombre_matriz)

    P = load_transport_matrix(path, one_indexed=True)
    _ = plot_eigs_complex(P, k=250, title=f"Espectro de autovalores: {mes_nombre.capitalize()} 2002")

    labels, feat, vals, vecs = spectral_clustering_from_P(
        P, n_vecs=6, n_clusters=6, random_state=SEED
    )
    plot_clusters_pca(feat, labels, title=f"Clustering espectral (PCA de autovectores) – {mes_nombre.capitalize()} 2002", vecs=vecs, vals=vals)
    
    
    
print("""
Cómo leer los colores (clusters)
Cada color corresponde a un cluster detectado por KMeans en el espacio espectral.
Los nodos del mismo color tienen autovectores similares,
es decir, dinámicas de transporte parecidas (pueden pertenecer a una misma región o comunidad del flujo).
Las fronteras entre colores son zonas donde el comportamiento cambia abruptamente
(por ejemplo, bordes entre cuencas, o regiones con distinta conectividad hidrodinámica).

Cómo interpretar la forma del gráfico
El patrón en forma de arco o “semicírculo” es típico de embeddings espectrales.
Los autovectores están correlacionados con modos oscilatorios o de mezcla.
La forma curva indica que existen modos graduales de transición,
no cortes abruptos entre comunidades.
Si los puntos formaran grupos compactos claramente separados, indicaría comunidades muy diferenciadas en la red de transporte.
Si se ven conectados (como un continuo), el sistema tiene zonas intermedias o mixtas.
""")


















































#%%


def calcular_entropia_estacionaria(P):
    """
    Calcula la entropía estacionaria del sistema a partir del autovector
    asociado al autovalor dominante λ=1 (estado estacionario).
    También entrega una interpretación automática del grado de mezcla.

    Por qué se usa P.T:
    - P describe transiciones "de i a j" por filas (P_ij).
    - El estado estacionario π cumple: π^T P = π^T
      → en forma matricial: P^T π = π
      Por eso se calcula el autovector derecho de P.T (o el izquierdo de P).
    """
    # calcular el autovector estacionario asociado al autovalor más grande (≈1)
    vals, vecs = eigs(P.T, k=1, which='LM')
    pi = np.abs(vecs[:, 0].real)
    pi /= pi.sum()  # normalizar para que sume 1

    # calcular entropía
    H = -np.sum(pi * np.log(pi + 1e-12))
    Hmax = np.log(len(pi))
    Hnorm = H / Hmax

    print(f"\n Entropía estacionaria: {H:.4f}")
    print(f" Entropía máxima posible (ln N): {Hmax:.4f}")
    print(f" Entropía normalizada (H/Hmax): {Hnorm:.4f}\n")

    # interpretación automática según rango
    if Hnorm > 0.85:
        interpretacion = "Sistema altamente mezclado (cercano a máxima entropía, flujo muy homogéneo)."
    elif Hnorm > 0.65:
        interpretacion = "Sistema parcialmente mezclado, con estructura leve en los flujos."
    elif Hnorm > 0.4:
        interpretacion = "Sistema con estructuras persistentes (regiones de flujo más concentradas)."
    elif Hnorm > 0.2:
        interpretacion = "Sistema fuertemente estructurado, presencia de regiones atrapantes o poco conectadas."
    else:
        interpretacion = "Sistema muy ordenado o direccional: el flujo sigue trayectorias casi deterministas."

    print(" Interpretación automática:")
    print(f"   {interpretacion}\n")
    print(" Nota: P se transpone porque el estado estacionario π cumple Pᵗ·π = π,")
    print("   es decir, las probabilidades estacionarias son autovectores izquierdos de P.\n")

    return H, Hmax, Hnorm






H, Hmax, Hnorm = calcular_entropia_estacionaria(P)


#%%

print(r"""
Relación matemática entre P, Pᵀ y P⁻¹

──────────────────────────────────────────────────────────────────────────────------------------------------------------------------------
│ Operación          │              Expresión matemática              │                             Significado                          │
├───────────---------┼────────────────────────────────────────────────┼──────────────────────────────------------------------------------┤
│ Evolución directa  │  xₜ₊₁ = P · xₜ                                  │  Paso temporal normal (dinámica hacia adelante).                 │
│ Proceso inverso    │  Pᵀ · π = π                                    │  Equilibrio o dinámica inversa (autovector estacionario).        │
│ Retroceso exacto   │  xₜ = P⁻¹ · xₜ₊₁                                │  Retroceso en el tiempo (sin interpretación probabilística).     │
│ Reversibilidad     │  D_π · P = Pᵀ · D_π                            │  Se cumple solo si el sistema es reversible (flujo equilibrado). │
│ Similitud espectral│ Pᵀ = D_π · P · D_π⁻¹                           │  P y Pᵀ tienen los mismos autovalores si hay reversibilidad.     │
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────-

Donde:
 - P  ∈ ℝⁿˣⁿ  es la matriz de transición (estocástica por filas).
 - π  es el vector estacionario tal que  Pᵀ·π = π.
 - D_π = diag(π)  es la matriz diagonal con π en la diagonal.
 - Si  D_π·P = Pᵀ·D_π,  el sistema cumple equilibrio detallado (reversible).
""")











#%% 
def analizar_propiedades_matriz(P):
    """
    Evalúa propiedades algebraicas de la matriz de transporte P:
    - Si es cuadrada
    - Si es invertible
    - Si es simétrica (P = Pᵀ)
    - Si es hermitiana (P = Pᴴ)
    - Si es dispersa (sparse)
    - Si es estocástica (filas suman 1)
    """

    print("\n================= INFORME DE PROPIEDADES DE LA MATRIZ P =================\n")
    print(f"Dimensiones de P: {P.shape[0]} x {P.shape[1]}")

    try:
        P_dense = P.toarray()
    except Exception:
        P_dense = np.array(P)

    if P.shape[0] == P.shape[1]:
        print("P es cuadrada (puede tener inversa si el determinante ≠ 0).")
    else:
        print("P no es cuadrada: no puede invertirse.")
        return

    is_symmetric = np.allclose(P_dense, P_dense.T, atol=1e-10)
    is_hermitian = np.allclose(P_dense, P_dense.conj().T, atol=1e-10)

    print(f"Es simétrica: {is_symmetric}")
    print(f"Es hermitiana (P = Pᴴ): {is_hermitian}")

    sparsity = 1.0 - (P.count_nonzero() / (P.shape[0] * P.shape[1]))
    print(f"Dispersión (proporción de ceros): {sparsity*100:.2f}%")

    row_sums = np.array(P.sum(axis=1)).ravel()
    stochastic_ok = np.allclose(row_sums, np.ones_like(row_sums), atol=1e-10)
    print(f"Estocástica por filas: {stochastic_ok}")

    print("\nIntentando calcular la inversa de P (puede fallar si es singular)...")
    try:
        from numpy.linalg import inv, LinAlgError
        P_inv = inv(P_dense)
        print("P es invertible (determinante ≠ 0).")
        det_val = np.linalg.det(P_dense)
        print(f"Determinante de P: {det_val:.6e}")
    except Exception as e:
        print(f"P no es invertible: {e}")
        P_inv = None

    try:
        vals = np.linalg.eigvals(P_dense)
        print(f"\nAutovalor máximo en magnitud: {np.max(np.abs(vals)):.4f}")
        print(f"Autovalor mínimo en magnitud: {np.min(np.abs(vals)):.4e}")
    except Exception:
        print("No se pudieron calcular autovalores (matriz muy grande o dispersa).")

    print("\n📘 Interpretación:")
    if is_hermitian:
        print("   → P es hermitiana: sus autovalores son reales y su base es ortogonal.")
    elif is_symmetric:
        print("   → P es simétrica (caso real de hermitiana).")
    else:
        print("   → P NO es hermitiana: puede tener autovalores complejos (sistema no reversible).")

    if stochastic_ok:
        print("   → P conserva probabilidad (cada fila suma 1).")
    else:
        print("   → P no es perfectamente estocástica (puede haber errores de normalización numérica).")

    if P_inv is None:
        print("   → No se puede invertir: el flujo no tiene trayectoria inversa única (sistema disipativo).")
    else:
        print("   → Se obtuvo P⁻¹ (interpretar solo algebraicamente, no como transición inversa).")

    print("\n=======================================================================\n")
    return P_inv

# Ejecutar el análisis
P_inv = analizar_propiedades_matriz(P)








print(r"""
INTERPRETACIÓN: MATRIZ DE TRANSPORTE NO INVERTIBLE (SINGULAR)

───────────────────────────────────────────────────────────────
Matemáticamente:
───────────────────────────────────────────────────────────────
- det(P) = 0   →  la matriz no tiene inversa.
- Existen filas o columnas linealmente dependientes.
- El rango de P es menor que su dimensión (rango(P) < N).
- El sistema P·x = y no tiene solución única (puede tener infinitas o ninguna).
- Existe un espacio nulo no trivial:  ∃ v ≠ 0 tal que  P·v = 0.

───────────────────────────────────────────────────────────────
Dinamismo y teoría de Markov:
───────────────────────────────────────────────────────────────
- El proceso no es reversible (no se puede reconstruir el estado previo).
- El sistema pierde información temporalmente (no hay dinámica uno a uno).
- La probabilidad puede concentrarse en subconjuntos del dominio.
- Existen estados absorbentes o regiones atrapantes.
- El equilibrio estacionario no es único (múltiples π posibles).
- Puede haber componentes comunicantes: subredes que no intercambian flujo.
- Algunos nodos nunca son visitados o dejan de recibir masa (columnas nulas).
- El flujo neto es disipativo (no conserva “información” bajo evolución inversa).

───────────────────────────────────────────────────────────────
Interpretación física (transporte oceánico, difusivo o ambiental):
───────────────────────────────────────────────────────────────
- Existen zonas atrapantes donde el flujo entra pero no sale.
- Existen zonas emisoras (fuentes) que expulsan masa sin recibirla.
- Parte de la masa puede “salir del dominio” o disiparse fuera del sistema.
- El transporte no es perfectamente conservativo (puede haber fuga o pérdida).
- Dos regiones pueden tener dinámicas indistinguibles (redundancia espacial).
- La red presenta caminos sin retorno (asimetría direccional del flujo).
- Puede existir conectividad parcial: el dominio se fragmenta en subcuencas.

───────────────────────────────────────────────────────────────
Consecuencias espectrales:
───────────────────────────────────────────────────────────────
- Al menos un autovalor es exactamente 0 → modo completamente disipativo.
- Autovalores con |λ| < 1 → relajaciones hacia estados absorbentes.
- El autovalor λ=1 puede tener multiplicidad >1 → múltiples equilibrios posibles.
- El espectro no está en el círculo unidad completo (no conservativo).

───────────────────────────────────────────────────────────────
Implicancias prácticas:
───────────────────────────────────────────────────────────────
- No se puede invertir P algebraicamente (no existe P⁻¹).
- No existe un proceso temporal “hacia atrás” único.
- Se requiere pseudo-inversa (Moore–Penrose) o análisis por subespacios.
- Los resultados de clustering o entropía deben interpretarse como dinámicas
  proyectadas o agregadas, no como flujos perfectamente reversibles.

───────────────────────────────────────────────────────────────
Resumen intuitivo:
───────────────────────────────────────────────────────────────
Una matriz de transporte no invertible representa un sistema:
→ con direccionalidad o disipación,
→ con pérdida de información o mezcla irreversible,
→ donde solo existe dinámica hacia adelante en el tiempo,
→ y donde el equilibrio puede no ser único o alcanzarse por varios caminos.
""")















#%%
def analizar_propiedades_transpuesta(P):
    """
    Analiza las propiedades algebraicas y estructurales de la traspuesta Pᵀ:
    - Simetría, hermiticidad, estocasticidad
    - Dispersión (sparsity)
    - Comparación con P (reversibilidad)
    - Interpretación física y dinámica

    Retorna la traspuesta Pᵀ (formato disperso si aplica).
    """

    print("\n================= ANÁLISIS DE LA MATRIZ TRASPUESTA (Pᵀ) =================")

    P_T = P.transpose().tocsr()
    N = P.shape[0]
    print(f"Dimensiones de P:  {P.shape[0]} × {P.shape[1]}")
    print(f"Dimensiones de Pᵀ: {P_T.shape[0]} × {P_T.shape[1]}")

    try:
        P_dense = P.toarray()
        P_T_dense = P_T.toarray()
    except Exception:
        print("Matriz muy grande; se analizará solo estructura dispersa.")
        P_dense = None
        P_T_dense = None

    if P_dense is not None:
        is_symmetric = np.allclose(P_dense, P_T_dense, atol=1e-10)
        is_hermitian = np.allclose(P_dense, P_dense.conj().T, atol=1e-10)
    else:
        is_symmetric = np.nan
        is_hermitian = np.nan

    print(f"P es simétrica (P = Pᵀ): {is_symmetric}")
    print(f"P es hermitiana (P = Pᴴ): {is_hermitian}")

    nnz = P_T.count_nonzero()
    sparsity = 1.0 - nnz / (P_T.shape[0] * P_T.shape[1])
    print(f"Dispersión (proporción de ceros): {sparsity*100:.2f}%")

    row_sums_T = np.array(P_T.sum(axis=1)).ravel()
    stochastic_T = np.allclose(row_sums_T, np.ones_like(row_sums_T), atol=1e-10)
    print(f"Estocástica por filas (Pᵀ): {stochastic_T}")

    if P_dense is not None and N < 800:
        eig_P = np.linalg.eigvals(P_dense)
        eig_PT = np.linalg.eigvals(P_T_dense)
        same_spectrum = np.allclose(np.sort_complex(eig_P), np.sort_complex(eig_PT), atol=1e-8)
        print(f"P y Pᵀ tienen los mismos autovalores: {same_spectrum}")
    else:
        same_spectrum = None
        print("No se comparó el espectro (matriz muy grande).")

    try:
        vals, vecs = eigs(P.T, k=1, which='LM')
        pi = np.abs(vecs[:,0].real)
        pi /= pi.sum()
        D_pi = np.diag(pi)
        left = D_pi @ P.toarray()
        right = P_T.toarray() @ D_pi
        reversible = np.allclose(left, right, atol=1e-6)
    except Exception:
        reversible = None
    print(f"Cumple equilibrio detallado (reversible): {reversible}")

    print("\nINTERPRETACIÓN:")
    if is_hermitian:
        print("→ P es hermitiana: dinámica completamente reversible, autovalores reales.")
    elif is_symmetric:
        print("→ P es simétrica: flujo bidireccional con igual probabilidad i↔j.")
    else:
        print("→ P NO es simétrica ni hermitiana: el sistema tiene direccionalidad o disipación.")

    if stochastic_T:
        print("→ Pᵀ conserva probabilidad (estocástica por filas).")
    else:
        print("→ Pᵀ no conserva masa exactamente (posible pérdida o acumulación).")

    if reversible:
        print("→ El sistema cumple equilibrio detallado: flujo hacia adelante y hacia atrás balanceado.")
    else:
        print("→ El sistema NO es reversible: existen trayectorias preferenciales o regiones atrapantes.")

    print("==========================================================================\n")

    return P_T

P_T = analizar_propiedades_transpuesta(P)
