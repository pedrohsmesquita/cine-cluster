from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
kmeans.fit(matriz_usuarios)
silhueta = silhouette_score(matriz_usuarios, kmeans.labels_)

def avaliar_k
(
    self,
        matriz_usuarios: pd.DataFrame,
        k_min: int = 2,
        k_max: int = 15,
        seed: int = 42,
) -> pd.DataFrame:
    """Avalia diferentes valores de k usando Silhueta e Cotovelo.
        Args:
            matriz_usuarios: DataFrame indexado por user_id com os 18 gêneros.
            k_min: Valor mínimo de clusters a ser testado (padrão: 2).
            k_max: Valor máximo de clusters a ser testado (padrão: 15).
            seed: Semente de aleatoriedade para reprodutibilidade.
        Returns:
            DataFrame com as colunas ['k', 'wcss', 'silhouette'].
        """
        metricas = []
        for k in range(k_min, k_max + 1):
            kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
            kmeans.fit(matriz_usuarios)
            wcss = kmeans.inertia_
            silhueta = silhouette_score(matriz_usuarios, kmeans.labels_)
            metricas.append({
                "k": k,
                "wcss": wcss,
                "silhouette": silhueta,
            })
        return pd.DataFrame(metricas)