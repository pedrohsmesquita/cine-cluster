import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

class CargaService:
    def __init__(
            self,
            conn: sqlite3.Connection,
            split_seed: int = 67,
            test_size: float = 0.2,
            val_seed: int = 11,
            val_size: float = 0.2,
        ):
            self.conn = conn
            self.conn.execute("PRAGMA foreign_keys = ON")
            self.split_seed = split_seed
            self.test_size = test_size
            self.val_seed = val_seed
            self.val_size = val_size

    def criar_schema(self, caminho_sql: str) -> None:
        """Executa o arquivo .sql que cria as tabelas.
    
        Args:
            caminho_sql: caminho do .sql relativo a onde se executa o programa python.
        """
        sql = Path(caminho_sql).read_text(encoding="utf-8")
        self.conn.executescript(sql)
        self.conn.execute("PRAGMA foreign_keys = ON")
    
    def _ler_dat(self, caminho: Path, colunas: list[str], **kwargs) -> pd.DataFrame:
        """Carrega um arquivo do dataset.
    
        Args:
            caminho: caminho do arquivo relativo a onde se executa o programa python.
            colunas: nome das colunas do dataset, já que não há cabeçalho.
            kwargs: opcional para estender a leitura do csv usando pandas.
    
        Returns:
            Um Panda DataFrame com o conteúdo do dataset em memória.
        """
        return pd.read_csv(
            caminho,
            sep="::",
            engine="python",
            encoding="latin-1",
            header=None,
            names=colunas,
            **kwargs,
        )
    
    def _dados_ja_carregados(self) -> bool:
        """Verifica se o banco de dados já está povoado."""
        return self.conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0] > 0
    
    def _extrair_generos_de_filme(self, filmes: pd.DataFrame) -> pd.DataFrame:
        """Extrai cada gênero de um filme como um par (filme, gênero).
    
        Args:
            filmes: tabela de filmes cujos gêneros são strings separadas por "|".
        """
        return (
            filmes[["movie_id", "genres"]]
            .assign(genre=lambda d: d["genres"].str.split("|"))
            .explode("genre")[["movie_id", "genre"]]
        )
    
    def _salvar_dados_no_banco(self,
                               filmes: pd.DataFrame,
                               movie_genres: pd.DataFrame,
                               usuarios: pd.DataFrame,
                               notas: pd.DataFrame) -> None:
        """Salva o conteúdo do dataset no Banco de Dados.
    
        Args:
            filmes: tabela de filmes em memória principal
            movie_genres: tabela com filmes e seus gêneros em memória principal
            usuarios: tabela de usuários em memória principal
            notas: tabela de avaliação dos filmes em memória principal
        """
        with self.conn:
            filmes[["movie_id", "title"]].to_sql(
                "movies", self.conn, if_exists="append", index=False
            )
            movie_genres.to_sql("movie_genres", self.conn, if_exists="append", index=False)
            usuarios.to_sql("users", self.conn, if_exists="append", index=False)
            notas.to_sql("ratings", self.conn, if_exists="append", index=False)
    
    def carregar_dados(self, pasta: str = "ml-1m") -> None:
        """Carrega no Banco de Dados todo o conteúdo do dataset.
    
        Args:
            pasta: caminho da pasta do dataset relativo a onde se executa o programa python.
        """
        pasta = Path(pasta)
    
        if self._dados_ja_carregados():
            print("Dados já carregados. Nada a fazer.")
            return
    
        filmes = self._ler_dat(pasta / "movies.dat", ["movie_id", "title", "genres"])
    
        movie_genres = self._extrair_generos_de_filme(filmes)
    
        usuarios = self._ler_dat(
            pasta / "users.dat",
            ["user_id", "gender", "age", "occupation", "zip_code"],
            dtype={"zip_code": str},
        )
    
        notas = self._ler_dat(
            pasta / "ratings.dat", ["user_id", "movie_id", "rating", "timestamp"]
        )
    
        self._salvar_dados_no_banco(filmes, movie_genres, usuarios, notas)
    
        print(f"Carregados: {len(filmes)} filmes, {len(usuarios)} usuários, {len(notas)} notas.")
    
    def _dados_ja_divididos(self) -> bool:
        """Verifica se os dados já passaram pelo processo de split.
    
        Returns:
            True, se a tabela de split já tiver sido povoada. False, caso contrário.
        """
        return self.conn.execute("SELECT COUNT(*) FROM split").fetchone()[0] > 0
    
    def _criar_tabela_split(self, treino: list, validacao: list, teste: list) -> None:
        """Cria a tabela em memória com dados rotulados para armazenagem no banco de dados.
    
        Args:
            treino: lista com ids dos usuários do conjunto treino
            validacao: lista com ids dos usuários do conjunto validação
            teste: lista com ids dos usuários do conjunto teste
        """
        split = pd.DataFrame(
            {
                "user_id": list(treino) + list(validacao) + list(teste),
                "subset": ["train"] * len(treino)
                + ["validation"] * len(validacao)
                + ["test"] * len(teste),
            }
        )
    
        with self.conn:
            split.to_sql("split", self.conn, if_exists="append", index=False)
    
    def criar_split(self) -> None:
        """Divide os usuários, deterministicamente, em treino/validação/teste e grava na tabela split.

        A primeira divisão separa o teste (test_size do total, com split_seed). A segunda
        separa a validação dentro do que sobrou (val_size do treino, com val_seed). Assim, o
        conjunto de teste não depende da validação.
        """
        if self._dados_ja_divididos():
            print("Split já foi realizado.")
            return
    
        user_ids = pd.read_sql("SELECT user_id FROM users ORDER BY user_id", self.conn)["user_id"]
        treino_total, teste = train_test_split(user_ids, test_size=self.test_size, random_state=self.split_seed)
        treino, validacao = train_test_split(treino_total, test_size=self.val_size, random_state=self.val_seed)
    
        self._criar_tabela_split(treino, validacao, teste)
    
        print(f"Split realizado: {len(treino)} treino, {len(validacao)} validação, {len(teste)} teste.")
    
    def carregar_notas(self, subsets: list[str]) -> pd.DataFrame:
        """Carrega do banco as notas dos usuários de um ou mais subconjuntos.
        
        Args:
            subsets: uma lista de 'train', 'validation' ou 'test' especificando o subconjunto
            a ser buscado.
        
        Returns:
            Tabela pandas com colunas user_id, movie_id e rating dos subconjuntos informados.
        """

        if not subsets or not set(subsets) <= {"train", "validation", "test"}:
            raise ValueError("subsets deve conter apenas 'train', 'validation' e/ou 'test'")

        placeholders = ",".join("?" * len(subsets))
        return pd.read_sql(
            f"""
            SELECT r.user_id, r.movie_id, r.rating
            FROM ratings r
            JOIN split s ON s.user_id = r.user_id
            WHERE s.subset IN ({placeholders})
            ORDER BY r.user_id, r.movie_id
            """,
            self.conn,
            params=tuple(subsets),
        )
    
    def carregar_generos_filmes(self) -> pd.DataFrame:
        """Carrega do banco um filme e seus gêneros.
        
        Returns:
            Tabela pandas com colunas movie_id e genre.
        """
        return pd.read_sql("SELECT movie_id, genre FROM movie_genres", self.conn)