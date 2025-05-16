import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

# Adicionei para se for sem pasta no seu computador ele adiciona ao rodar o codigo
os.makedirs("grafos_im", exist_ok=True)

def grafo(G, titulo, nome_arquivo, grupos=None):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(6, 4))
    if grupos:
        cores = []
        for no in G.nodes():
            if no in grupos[0]:
                cores.append('red')
            else:
                cores.append('green')
    else:
        cores = 'lightblue'
    nx.draw(G, pos, with_labels=True, node_color=cores, edge_color='gray')
    plt.title(titulo)
    plt.tight_layout()
    plt.savefig(f"grafos_im/{nome_arquivo}.png")
    plt.close()

def analise_espectral(G):
    A = nx.to_numpy_array(G)
    graus = np.sum(A, axis=1)
    D = np.diag(graus)
    L = D - A #Laplaciana
    autovalores, autovetores = np.linalg.eigh(L)

    num_componentes = np.sum(np.isclose(autovalores, 0))
    fiedler = autovetores[:, 1]
    grupo_vermelho = []
    grupo_verde = []
    for i in range(len(fiedler)):
        if fiedler[i] <= 0:
            grupo_vermelho.append(i)
        else:
            grupo_verde.append(i)
    return autovalores, num_componentes, (grupo_vermelho, grupo_verde)

def grafos_im():
    grafos = []

    G1 = nx.Graph()
    G1.add_edges_from([
        (0, 1), (0, 2), (1, 2), (1, 3),
        (3, 4), (4, 5), (5, 6), (6, 7), (5, 7)
    ])
    grafos.append(("Grafo 1", G1))

    G2 = nx.Graph()
    G2.add_edges_from([
        (0, 1), (1, 2), (2, 3),
        (4, 5), (5, 6), (6, 7),
        (3, 4)
    ])
    grafos.append(("Grafo 2", G2))

    G3 = nx.complete_graph(8)
    grafos.append(("Grafo 3", G3))

    espectros = []

    for i, (nome, G) in enumerate(grafos, start=1):
        autovalores, componentes, grupos = analise_espectral(G)
        print(f"{nome} - Autovalores:")
        print(np.round(autovalores, 4))
        print(f"Componentes conexas detectadas: {componentes}\n")
        grafo(G, f"{nome} - Separação pelo vetor de Fiedler", f"grafo{i}_fiedler", grupos)
        espectros.append(autovalores)

    plt.figure(figsize=(8, 4))
    for i, autovalores in enumerate(espectros):
        plt.plot(sorted(autovalores), marker='o', label=f"Grafo {i+1}")
    plt.title("Espectro dos autovalores (Matriz Laplaciana)")
    plt.xlabel("Índice do autovalor")
    plt.ylabel("Valor")
    plt.legend()
    plt.tight_layout()
    plt.savefig("grafos_im/espectros_laplacianos.png")
    plt.close()

if __name__ == "__main__":
    grafos_im()
