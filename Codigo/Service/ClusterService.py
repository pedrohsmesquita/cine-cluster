import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class ClusterService:
    def avaliar_k(
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

    def treinar(
        self,
        matriz_usuarios: pd.DataFrame,
        k: int,
        seed: int = 42,
    ) -> tuple[KMeans, pd.DataFrame, pd.DataFrame]:
        """Treina o modelo K-Means final e extrai as estruturas para o banco de dados.

        A partir da matriz de notas por gênero, ajusta os centróides para o número 'k'
        de grupos definido, mapeia a qual cluster cada usuário pertence e calcula o
        centróide de cada grupo junto com a quantidade de usuários associada.

        Args:
            matriz_usuarios: DataFrame indexado por 'user_id' com as 18 colunas de gêneros
                preenchidas com as notas médias.
            k: Quantidade de clusters escolhida para o agrupamento.
            seed: Semente aleatória para garantir que o resultado seja reproduzível.

        Returns:
            Uma tupla contendo três elementos:
            - kmeans: A instância do objeto KMeans do scikit-learn já ajustada (.fit).
            - cluster_usuarios: DataFrame com colunas ['user_id', 'cluster_id'].
            - df_centroides: DataFrame com colunas ['cluster_id', 'n_users'] seguidas das
              18 colunas dos gêneros (formatadas em minúsculo e sem caracteres especiais)
              contendo as coordenadas do centro de cada cluster.
        """
        # 1. Instancia o algoritmo com o k escolhido, semente fixa e 10 inicializações
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)

        # 2. Executa o agrupamento (ajuste dos centróides) com base nos dados dos usuários
        kmeans.fit(matriz_usuarios)

        # 3. Cria a relação direta entre o ID do usuário e o cluster atribuído a ele
        cluster_usuarios = pd.DataFrame({
            "user_id": matriz_usuarios.index,
            "cluster_id": kmeans.labels_,
        })

        # 4. Converte os centróides numéricos gerados pelo scikit-learn em DataFrame
        df_centroides = pd.DataFrame(
            kmeans.cluster_centers_,
            columns=matriz_usuarios.columns,
        )

        # 5. Adiciona o identificador do cluster (0, 1, 2, ... k-1)
        df_centroides["cluster_id"] = df_centroides.index

        # 6. Conta quantos usuários caíram em cada cluster e adiciona na coluna 'n_users'
        df_centroides["n_users"] = df_centroides["cluster_id"].map(
            cluster_usuarios["cluster_id"].value_counts()
        )

        # 7. Formata os nomes das colunas para letras minúsculas e sem caracteres especiais
        #    (ex: "Film-Noir" -> "film_noir", "Children's" -> "childrens") para coincidir com o schema SQL
        df_centroides.columns = [
            col.lower().replace("-", "_").replace("'", "")
            for col in df_centroides.columns
        ]

        return kmeans, cluster_usuarios, df_centroides