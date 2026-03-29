# -*- coding: utf-8 -*-
"""
Solver Exato: Programação Linear Inteira (PLI) via PuLP + CBC.

─────────────────────────────────────────────────────────────────────────────
FORMULAÇÃO MATEMÁTICA DO PROBLEMA
─────────────────────────────────────────────────────────────────────────────

Dados:
    n              → número de projetos candidatos
    lucro_i        → lucro esperado do projeto i
    custo_i        → custo em R$ do projeto i
    horas_i        → horas de trabalho necessárias para o projeto i
    B              → orçamento máximo disponível
    H              → horas máximas disponíveis

Variáveis de decisão:
    x_i ∈ {0, 1}   →  1 se o projeto i é selecionado, 0 caso contrário

Função objetivo (maximizar):
    max  Σᵢ lucro_i · x_i

Restrições:
    Σᵢ custo_i · x_i  ≤  B     (orçamento não pode ser excedido)
    Σᵢ horas_i · x_i  ≤  H     (horas não podem ser excedidas)
    x_i ∈ {0, 1}      ∀i       (decisão binária: executa ou não)

─────────────────────────────────────────────────────────────────────────────
Este é um caso especial da Mochila 0-1 Multidimensional.
O solver CBC (Coin-or Branch and Cut) usa Branch & Bound com relaxação LP
para encontrar a solução GLOBALMENTE ÓTIMA com certificado de otimalidade.
─────────────────────────────────────────────────────────────────────────────
"""

import time

import pandas as pd
import pulp


def solve_exact(
    df: pd.DataFrame,
    constraints: dict,
    verbose: bool = False,
) -> dict:
    """
    Resolve o problema de alocação usando Programação Linear Inteira (PLI).

    Parâmetros:
        df          : DataFrame de projetos (colunas: custo, horas, lucro)
        constraints : dict com 'orcamento_maximo' e 'horas_maximas'
        verbose     : se True, exibe o log do solver CBC

    Retorna:
        dict com:
            metodo               → nome do método
            projetos_selecionados → lista de índices (0-based) selecionados
            n_selecionados        → quantidade de projetos selecionados
            lucro_total           → lucro total obtido (R$)
            custo_usado           → orçamento utilizado (R$)
            horas_usadas          → horas utilizadas
            tempo_execucao        → tempo em segundos
            status                → status do solver ('Optimal', etc.)
    """
    inicio = time.perf_counter()

    n = len(df)

    # ── 1. Criar o modelo de Programação Linear Inteira ───────────────────
    modelo = pulp.LpProblem("Alocacao_de_Recursos", pulp.LpMaximize)

    # ── 2. Variáveis de decisão: x[i] ∈ {0, 1} ───────────────────────────
    # LpVariable com cat='Binary' cria automaticamente x_i ∈ {0, 1}
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

    # ── 3. Função objetivo: maximizar lucro total ─────────────────────────
    modelo += (
        pulp.lpSum(int(df["lucro"].iloc[i]) * x[i] for i in range(n)),
        "Maximizar_Lucro_Total",
    )

    # ── 4. Restrição de orçamento ─────────────────────────────────────────
    modelo += (
        pulp.lpSum(int(df["custo"].iloc[i]) * x[i] for i in range(n))
        <= constraints["orcamento_maximo"],
        "Restricao_Orcamento",
    )

    # ── 5. Restrição de horas da equipe ───────────────────────────────────
    modelo += (
        pulp.lpSum(int(df["horas"].iloc[i]) * x[i] for i in range(n))
        <= constraints["horas_maximas"],
        "Restricao_Horas",
    )

    # ── 6. Resolver com o solver CBC (incluído no PuLP) ───────────────────
    # msg=0 suprime a saída verbosa do solver (a menos que verbose=True)
    solver = pulp.PULP_CBC_CMD(msg=1 if verbose else 0)
    status_code = modelo.solve(solver)

    tempo = time.perf_counter() - inicio

    # ── 7. Extrair e retornar resultados ──────────────────────────────────
    # pulp.value(x[i]) retorna 1.0 se o projeto foi selecionado
    selecionados = [i for i in range(n) if pulp.value(x[i]) is not None and pulp.value(x[i]) > 0.5]

    lucro_total = sum(df["lucro"].iloc[i] for i in selecionados)
    custo_usado = sum(df["custo"].iloc[i] for i in selecionados)
    horas_usadas = sum(df["horas"].iloc[i] for i in selecionados)

    return {
        "metodo": "Exato (PLI - PuLP/CBC)",
        "projetos_selecionados": selecionados,
        "n_selecionados": len(selecionados),
        "lucro_total": float(lucro_total),
        "custo_usado": float(custo_usado),
        "horas_usadas": float(horas_usadas),
        "tempo_execucao": tempo,
        "status": pulp.LpStatus[status_code],
        "gap": 0.0,  # solução exata → gap de otimalidade é 0
    }
