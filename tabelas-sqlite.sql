PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    user_id     INTEGER PRIMARY KEY,
    gender      TEXT    NOT NULL CHECK (gender IN ('M','F')),
    age         INTEGER NOT NULL,
    occupation  INTEGER NOT NULL,
    zip_code    TEXT
);

CREATE TABLE IF NOT EXISTS movies (
    movie_id    INTEGER PRIMARY KEY,
    title       TEXT    NOT NULL
);

-- Essa tabela é derivada da informação do filme e de seus gêneros
CREATE TABLE IF NOT EXISTS movie_genres (
    movie_id    INTEGER NOT NULL REFERENCES movies(movie_id),
    genre       TEXT    NOT NULL,
    PRIMARY KEY (movie_id, genre)
);

CREATE INDEX IF NOT EXISTS idx_mg_genre ON movie_genres(genre);

CREATE TABLE IF NOT EXISTS ratings (
    user_id     INTEGER NOT NULL REFERENCES users(user_id),
    movie_id    INTEGER NOT NULL REFERENCES movies(movie_id),
    rating      INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    timestamp   INTEGER NOT NULL,
    PRIMARY KEY (user_id, movie_id)
);
CREATE INDEX IF NOT EXISTS idx_ratings_movie ON ratings(movie_id);

-- ===== Experimentos =====
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at     TEXT    NOT NULL,
    seed           INTEGER NOT NULL,
    k_clusters     INTEGER NOT NULL,
    min_ratings    INTEGER NOT NULL,
    model_path     TEXT,
    fill_zero INTEGER NOT NULL CHECK (fill_zero IN (0, 1)),
    centered INTEGER NOT NULL CHECK (centered IN (0, 1))
);

CREATE TABLE IF NOT EXISTS split (
    user_id        INTEGER PRIMARY KEY REFERENCES users(user_id),
    subset         TEXT NOT NULL CHECK (subset IN ('train', 'validation', 'test'))
);

-- ===== Resultados do treino (todos ligados a um experimento) =====
-- k_selection descreve as metricas de silhueta e cotovela para cada valor de k
-- para um experimento. Do ponto de vista da aplicação, primeiro deve ser
-- determinado todos esses valores e o k ser escolhido, depois o registro em
-- experiments deve ser criado (com o melhor k) e, por fim, deve ser registrado
-- em k_selection os valores da silhueta e do wcss para cada k referente ao
-- experimento criado, permitindo traçar o gráfico de silhueta e cotovelo.
CREATE TABLE IF NOT EXISTS k_selection (
    experiment_id  INTEGER NOT NULL REFERENCES experiments(experiment_id),
    k              INTEGER NOT NULL,
    silhouette     REAL    NOT NULL,
    wcss           REAL    NOT NULL,
    PRIMARY KEY (experiment_id, k)
);

CREATE TABLE IF NOT EXISTS user_cluster (
    experiment_id  INTEGER NOT NULL REFERENCES experiments(experiment_id),
    user_id        INTEGER NOT NULL REFERENCES users(user_id),
    cluster_id     INTEGER NOT NULL,
    PRIMARY KEY (experiment_id, user_id)
);

CREATE TABLE IF NOT EXISTS centroids (
    experiment_id  INTEGER NOT NULL REFERENCES experiments(experiment_id),
    cluster_id     INTEGER NOT NULL,
    n_users        INTEGER NOT NULL,
    action REAL, adventure REAL, animation REAL, childrens REAL, comedy REAL,
    crime REAL, documentary REAL, drama REAL, fantasy REAL, film_noir REAL,
    horror REAL, musical REAL, mystery REAL, romance REAL, sci_fi REAL,
    thriller REAL, war REAL, western REAL,
    PRIMARY KEY (experiment_id, cluster_id)
);

CREATE TABLE IF NOT EXISTS movie_cluster_stats (
    experiment_id  INTEGER NOT NULL REFERENCES experiments(experiment_id),
    movie_id       INTEGER NOT NULL REFERENCES movies(movie_id),
    cluster_id     INTEGER NOT NULL,
    avg_rating     REAL    NOT NULL,
    n_ratings      INTEGER NOT NULL,
    PRIMARY KEY (experiment_id, movie_id, cluster_id)
);

-- ===== Avaliação =====
CREATE TABLE IF NOT EXISTS results (
    result_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   INTEGER NOT NULL REFERENCES experiments(experiment_id),
    method          TEXT    NOT NULL,   -- 'kmeans' - estimativa da nota do filme veio do cluster. 'baseline_media_filme' - estimativa da nota do filme veio do conjunto treino inteiro. hipótese: kmeans precisa ter um mae menor.
    eval_subset TEXT NOT NULL CHECK (eval_subset IN ('validation', 'test')),
    mae             REAL,
    coverage        REAL,           --  Porcentagem das notas escondidas que o sistema conseguiu dar predizer. É uma métrica para não deixar a média do erro ser o único peso de avaliação do modelo.
    n_users         INTEGER,
    n_ratings       INTEGER,
    mae_common  REAL,      -- MAE só nas notas que os dois métodos estimaram
    n_common    INTEGER,   -- tamanho dessa interseção (igual nas duas linhas)
    UNIQUE (experiment_id, method, eval_subset)
);