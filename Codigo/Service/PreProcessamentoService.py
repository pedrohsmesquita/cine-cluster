import pandas as pd

GENEROS = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

class PreProcessamentoService:
    def __init__(
        self,
        hide_seed: int = 7,
        hide_frac: float = 0.3,
    ):
        self.hide_seed = hide_seed
        self.hide_frac = hide_frac


    def dividir_visivel_escondido(
        self, notas_teste: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Para cada usuário de teste, esconde 'hide_frac' porcento das notas. Utilizado para validar modelo.

        Args:
            notas_teste: tabela pandas das notas dadas por um usuário a um filme

        Returns:
            Tupla com as notas visíveis para prever classificação e notas escondidas para validar modelo.
        """
        notas_teste = notas_teste.sort_values(["user_id", "movie_id"])
        escondidas = notas_teste.groupby("user_id").sample(
            frac=self.hide_frac, random_state=self.hide_seed
        )
        visiveis = notas_teste.drop(escondidas.index)
        return visiveis, escondidas

    def matriz_generos(
        self,
        notas: pd.DataFrame,
        generos_filmes: pd.DataFrame,
        preencher_com_zero: bool = False
    ) -> pd.DataFrame:
        """Calcula a nota média por gênero de filme para cada usuário.

        Permite escolher a estratégia de tratamento para gêneros não avaliados:
        preenchimento com o valor neutro 0.0 ou com a média geral de avaliações
        dadas pelo próprio usuário.

        Args:
            notas: Tabela com colunas ['user_id', 'movie_id', 'rating'].
            generos_filmes: Tabela com pares ['movie_id', 'genre'].
            preencher_com_zero: Se True, preenche gêneros não avaliados com 0.0.
                Se False, preenche com a média geral de notas do próprio usuário
                (padrão: False).

        Returns:
            DataFrame indexado por 'user_id', contendo 18 colunas correspondentes
            aos gêneros e as respectivas notas médias de cada usuário.
        """
        notas_com_genero = notas.merge(generos_filmes, on="movie_id")
        
        matriz = notas_com_genero.pivot_table(
            index="user_id",
            columns="genre",
            values="rating",
            aggfunc="mean",
        )
        matriz = matriz.reindex(columns=GENEROS)

        if preencher_com_zero:
            matriz_preenchida = matriz.fillna(0.0)
        else:
            media_geral_usuario = notas.groupby("user_id")["rating"].mean()
            matriz_preenchida = matriz.T.fillna(media_geral_usuario).T

        return matriz_preenchida