import pandas as pd
from sklearn.cluster import KMeans


class RecomendadorService:
    def __init__(self, min_ratings: int = 5):
        self.min_ratings = min_ratings

    def calcular_estatisticas_cluster(
        self,
        notas_treino: pd.DataFrame,
        cluster_usuarios_treino: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calcula a nota média e a quantidade de avaliações de cada filme dentro de cada cluster.

        Args:
            notas_treino: DataFrame com colunas ['user_id', 'movie_id', 'rating'].
            cluster_usuarios_treino: DataFrame com colunas ['user_id', 'cluster_id'].

        Returns:
            DataFrame com colunas ['movie_id', 'cluster_id', 'avg_rating', 'n_ratings'].
        """
        notas_com_cluster = notas_treino.merge(cluster_usuarios_treino, on="user_id")

        stats = (
            notas_com_cluster.groupby(["movie_id", "cluster_id"], as_index=False)
            .agg(
                avg_rating=("rating", "mean"),
                n_ratings=("rating", "count"),
            )
        )
        return stats

    def prever_clusters(
        self,
        kmeans: KMeans,
        matriz_usuarios: pd.DataFrame,
    ) -> pd.DataFrame:
        """Descobre a qual cluster pertencem os usuários de teste a partir de suas notas visíveis.

        Args:
            kmeans: Modelo KMeans já treinado.
            matriz_usuarios: DataFrame indexado por 'user_id' com os 18 gêneros.

        Returns:
            DataFrame com colunas ['user_id', 'cluster_id'].
        """
        return pd.DataFrame({
            "user_id": matriz_usuarios.index,
            "cluster_id": kmeans.predict(matriz_usuarios),
        })

    def recomendar(
        self,
        user_id: int,
        cluster_id: int,
        notas_visiveis_usuario: pd.DataFrame,
        stats_cluster: pd.DataFrame,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """Recomenda os Top-N filmes mais bem avaliados no cluster que o usuário ainda não viu.

        Args:
            user_id: ID do usuário.
            cluster_id: Cluster ao qual o usuário pertence.
            notas_visiveis_usuario: DataFrame com as avaliações conhecidas ['user_id', 'movie_id', 'rating'].
            stats_cluster: DataFrame retornado por calcular_estatisticas_cluster.
            top_n: Quantidade de filmes para recomendar.

        Returns:
            DataFrame com ['movie_id', 'cluster_id', 'avg_rating', 'n_ratings'] dos filmes recomendados.
        """
        assistidos = set(
            notas_visiveis_usuario.loc[
                notas_visiveis_usuario["user_id"] == user_id, "movie_id"
            ]
        )

        candidatos = stats_cluster[
            (stats_cluster["cluster_id"] == cluster_id)
            & (stats_cluster["n_ratings"] >= self.min_ratings)
            & (~stats_cluster["movie_id"].isin(assistidos))
        ]

        return candidatos.sort_values(
            by=["avg_rating", "n_ratings"], ascending=[False, False]
        ).head(top_n)

    def avaliar(
        self,
        notas_treino: pd.DataFrame,
        notas_escondidas: pd.DataFrame,
        clusters_teste: pd.DataFrame,
        stats_cluster: pd.DataFrame,
    ) -> pd.DataFrame:
        """Avalia o erro médio absoluto (MAE) e cobertura (coverage) nas notas escondidas do teste.

        Compara o método 'kmeans' (média do filme no cluster) contra o
        'baseline_media_filme' (média geral do filme em todo o conjunto de treino).

        Returns:
            DataFrame com colunas ['method', 'mae', 'coverage', 'n_users', 'n_ratings'].
        """
        total_escondidas = len(notas_escondidas)
        n_usuarios = notas_escondidas["user_id"].nunique()

        # 1. Estimativa pelo K-Means (média do filme dentro do cluster do usuário)
        stats_validos = stats_cluster[stats_cluster["n_ratings"] >= self.min_ratings]
        pred_kmeans = (
            notas_escondidas.merge(clusters_teste, on="user_id")
            .merge(stats_validos, on=["movie_id", "cluster_id"], how="left")
            .dropna(subset=["avg_rating"])
        )
        mae_kmeans = (pred_kmeans["rating"] - pred_kmeans["avg_rating"]).abs().mean()
        cov_kmeans = len(pred_kmeans) / total_escondidas if total_escondidas > 0 else 0.0

        # 2. Estimativa pelo Baseline (média geral do filme no treino inteiro)
        media_filme_treino = (
            notas_treino.groupby("movie_id", as_index=False)
            .agg(avg_rating=("rating", "mean"), n_ratings=("rating", "count"))
        )
        media_valida = media_filme_treino[media_filme_treino["n_ratings"] >= self.min_ratings]
        pred_base = (
            notas_escondidas.merge(media_valida, on="movie_id", how="left")
            .dropna(subset=["avg_rating"])
        )
        mae_base = (pred_base["rating"] - pred_base["avg_rating"]).abs().mean()
        cov_base = len(pred_base) / total_escondidas if total_escondidas > 0 else 0.0

        # 3. Interseção: notas que os DOIS métodos conseguiram estimar
        chaves = ["user_id", "movie_id"]
        comuns = pred_kmeans[chaves].merge(pred_base[chaves], on=chaves)

        k_comum = pred_kmeans.merge(comuns, on=chaves)
        b_comum = pred_base.merge(comuns, on=chaves)

        mae_kmeans_comum = (k_comum["rating"] - k_comum["avg_rating"]).abs().mean()
        mae_base_comum = (b_comum["rating"] - b_comum["avg_rating"]).abs().mean()
        n_comum = len(comuns)

        return pd.DataFrame([
            {
                "method": "kmeans",
                "mae": float(mae_kmeans),
                "mae_common": float(mae_kmeans_comum),
                "coverage": float(cov_kmeans),
                "n_users": int(n_usuarios),
                "n_ratings": int(len(pred_kmeans)),
                "n_common": int(n_comum),
            },
            {
                "method": "baseline_media_filme",
                "mae": float(mae_base),
                "mae_common": float(mae_base_comum),
                "coverage": float(cov_base),
                "n_users": int(n_usuarios),
                "n_ratings": int(len(pred_base)),
                "n_common": int(n_comum),
            },
        ])