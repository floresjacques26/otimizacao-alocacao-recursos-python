# -*- coding: utf-8 -*-
"""
Módulo de geração de dados simulados para o problema de alocação de recursos.

Contexto do problema:
    Uma empresa precisa decidir quais projetos executar dado que tem:
      - Um orçamento limitado (em R$)
      - Uma equipe com disponibilidade limitada de horas

    Cada projeto candidato possui:
      - custo:  valor em R$ necessário para executar o projeto
      - horas:  horas de trabalho da equipe necessárias
      - lucro:  retorno financeiro esperado se o projeto for executado

    Objetivo: selecionar o subconjunto de projetos que MAXIMIZA o lucro total
    sem ultrapassar o orçamento nem as horas disponíveis.

    Este é um caso especial da Mochila Multidimensional (Multi-Dimensional Knapsack),
    um problema NP-difícil clássico em otimização combinatória.
"""

import numpy as np
import pandas as pd


def generate_projects(
    n_projects: int,
    custo_range: tuple = (10, 100),
    horas_range: tuple = (5, 50),
    lucro_range: tuple = (20, 200),
    seed: int = 42,
) -> pd.DataFrame:
    """
    Gera um DataFrame com dados simulados de projetos candidatos.

    Parâmetros:
        n_projects   : número de projetos a gerar
        custo_range  : (min, max) do custo de cada projeto em R$
        horas_range  : (min, max) das horas necessárias por projeto
        lucro_range  : (min, max) do lucro esperado por projeto
        seed         : semente aleatória para reprodutibilidade

    Retorna:
        pd.DataFrame com colunas:
            projeto_id  → identificador único (int)
            nome        → nome legível (string)
            custo       → custo em R$ (int)
            horas       → horas de trabalho necessárias (int)
            lucro       → lucro esperado em R$ (int)
            eficiencia  → lucro / (custo + horas) — razão de valor por recurso
    """
    rng = np.random.default_rng(seed)

    dados = {
        "projeto_id": range(1, n_projects + 1),
        "nome": [f"Proj_{i:03d}" for i in range(1, n_projects + 1)],
        "custo": rng.integers(custo_range[0], custo_range[1] + 1, size=n_projects),
        "horas": rng.integers(horas_range[0], horas_range[1] + 1, size=n_projects),
        "lucro": rng.integers(lucro_range[0], lucro_range[1] + 1, size=n_projects),
    }

    df = pd.DataFrame(dados)

    # Eficiência: lucro por unidade de recurso total consumido
    # Usada pelo algoritmo guloso como critério de priorização
    df["eficiencia"] = df["lucro"] / (df["custo"] + df["horas"])

    return df.reset_index(drop=True)


def generate_constraints(df: pd.DataFrame, utilizacao: float = 0.55) -> dict:
    """
    Gera as restrições de orçamento e horas baseadas na soma total dos projetos.

    A ideia é que o limite permita selecionar ~55% dos recursos totais,
    criando um problema não-trivial (nem tudo cabe, mas muita coisa cabe).

    Parâmetros:
        df          : DataFrame de projetos gerado por generate_projects()
        utilizacao  : fração dos recursos totais disponível (padrão: 0.55)

    Retorna:
        dict com chaves:
            orcamento_maximo → orçamento disponível em R$
            horas_maximas    → horas disponíveis da equipe
    """
    orcamento = int(df["custo"].sum() * utilizacao)
    horas = int(df["horas"].sum() * utilizacao)

    return {
        "orcamento_maximo": orcamento,
        "horas_maximas": horas,
    }


def get_scenarios() -> dict:
    """
    Define três cenários com tamanhos crescentes para análise comparativa.

    Cenários:
        pequeno  →  10 projetos  (instância trivial para o solver exato)
        medio    →  30 projetos  (instância moderada)
        grande   → 100 projetos  (instância que começa a desafiar o solver exato)

    Retorna:
        dict mapeando nome do cenário → {n_projetos, descricao}
    """
    return {
        "pequeno": {
            "n_projetos": 10,
            "descricao": "Pequena empresa — 10 projetos candidatos",
        },
        "medio": {
            "n_projetos": 30,
            "descricao": "Média empresa — 30 projetos candidatos",
        },
        "grande": {
            "n_projetos": 100,
            "descricao": "Grande empresa — 100 projetos candidatos",
        },
    }
