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
        preencher_com_zero: bool = False,
        centralizar: bool = False,
    ) -> pd.DataFrame:
        """Calcula a nota média por gênero de filme para cada usuário.

        Permite configurar o pré-processamento para testar diferentes hipóteses:
        - Estratégia de preenchimento para gêneros não avaliados (0.0 vs média do usuário).
        - Centralização pela média geral do usuário (notas relativas). Quando 'centralizar' é
        True, o preenchimento deixa de importar, pois os não avaliados serão 0 (neutro).

        Args:
            notas: Tabela com colunas ['user_id', 'movie_id', 'rating'].
            generos_filmes: Tabela com pares ['movie_id', 'genre'].
            preencher_com_zero: Se True, preenche gêneros ausentes com 0.0 puro.
                Se False, usa a média de avaliações do próprio usuário (padrão: False).
            centralizar: Se True, subtrai a média geral do usuário de todas as notas,
                transformando a escala em preferências relativas (padrão: False).

        Returns:
            DataFrame indexado por 'user_id' com 18 colunas correspondentes aos gêneros.
        """
        # 1. Cruzar notas com gêneros e criar a grade (usuário x gênero)
        notas_com_genero = notas.merge(generos_filmes, on="movie_id")
        matriz = notas_com_genero.pivot_table(
            index="user_id",
            columns="genre",
            values="rating",
            aggfunc="mean",
        )
        matriz = matriz.reindex(columns=GENEROS)

        # 2. Calcular a média geral de cada usuário (caso seja necessária)
        media_geral_usuario = notas.groupby("user_id")["rating"].mean()

        # 3. Centralizar pela média do usuário (se solicitado)
        if centralizar:
            return matriz.sub(media_geral_usuario, axis=0).fillna(0.0)

        # 4. Tratar valores ausentes (NaN)
        if preencher_com_zero:
            return matriz.fillna(0.0)

        return matriz.where(matriz.notna(), media_geral_usuario, axis=0)