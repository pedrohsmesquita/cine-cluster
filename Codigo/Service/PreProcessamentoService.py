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

    # TODO: Implementar processamento das notas médias por gênero de filme para
    #       cada usuário. Ainda precisa ser considerado como lidar nas situações
    #       em que o usuário avaliou poucos filmes de um mesmo gênero, pois isso
    #       pode inflar a média daquele gênero.
    def vetor_genero(
        self,
        notas: pd.DataFrame,
        generos_filmes: pd.DataFrame,
    ) -> pd.DataFrame:
        pass