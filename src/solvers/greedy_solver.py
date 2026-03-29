"""
Solver Heurístico: Algoritmo Guloso (Greedy).

─────────────────────────────────────────────────────────────────────────────
IDEIA DO ALGORITMO
─────────────────────────────────────────────────────────────────────────────

Estratégia:
    1. Calcular uma pontuação de prioridade para cada projeto
    2. Ordenar os projetos da maior para a menor pontuação
    3. Percorrer a lista em ordem e selecionar o projeto se ele couber
       dentro das restrições restantes; pular caso contrário

Critérios de priorização disponíveis:
    'eficiencia' → lucro / (custo + horas)  [padrão — melhor resultado geral]
    'lucro'      → lucro bruto (pode gastar muito recurso por pouco adicional)
    'custo'      → projetos mais baratos primeiro (ignora horas e lucro)

Complexidade:
    O(n log n) — dominado pela ordenação
    Extremamente rápido, mas NÃO garante solução ótima.

Exemplo de falha do greedy:
    Projetos: A(custo=10, lucro=15), B(custo=6, lucro=9), C(custo=6, lucro=9)
    Orçamento: 12
    Greedy escolhe A (eficiência 1.5), mas B+C dão lucro 18 > 15.
─────────────────────────────────────────────────────────────────────────────
"""

import time

import pandas as pd


def solve_greedy(
    df: pd.DataFrame,
    constraints: dict,
    criterion: str = "eficiencia",
) -> dict:
    """
    Resolve o problema de alocação com algoritmo guloso.

    Parâmetros:
        df          : DataFrame de projetos (colunas: custo, horas, lucro, eficiencia)
        constraints : dict com 'orcamento_maximo' e 'horas_maximas'
        criterion   : critério de ordenação — 'eficiencia', 'lucro' ou 'custo'

    Retorna:
        dict com campos compatíveis com o solver exato para comparação direta.
    """
    inicio = time.perf_counter()

    # Rastrear recursos ainda disponíveis
    orcamento_restante = constraints["orcamento_maximo"]
    horas_restantes = constraints["horas_maximas"]

    # ── Passo 1: Ordenar por critério de priorização ──────────────────────
    if criterion == "eficiencia":
        # Melhor razão lucro por recurso consumido
        df_ordenado = df.sort_values("eficiencia", ascending=False)
    elif criterion == "lucro":
        # Maior lucro bruto primeiro
        df_ordenado = df.sort_values("lucro", ascending=False)
    elif criterion == "custo":
        # Menor custo primeiro (para caber mais projetos)
        df_ordenado = df.sort_values("custo", ascending=True)
    else:
        raise ValueError(f"Critério inválido: '{criterion}'. Use 'eficiencia', 'lucro' ou 'custo'.")

    # ── Passo 2: Seleção gulosa ───────────────────────────────────────────
    selecionados = []

    for _, projeto in df_ordenado.iterrows():
        # Verificar se o projeto cabe dentro dos recursos restantes
        cabe_no_orcamento = projeto["custo"] <= orcamento_restante
        cabe_nas_horas = projeto["horas"] <= horas_restantes

        if cabe_no_orcamento and cabe_nas_horas:
            selecionados.append(projeto.name)  # .name = índice original do DataFrame
            orcamento_restante -= projeto["custo"]
            horas_restantes -= projeto["horas"]

    tempo = time.perf_counter() - inicio

    # ── Calcular métricas do resultado ────────────────────────────────────
    lucro_total = float(df.loc[selecionados, "lucro"].sum())
    custo_usado = float(df.loc[selecionados, "custo"].sum())
    horas_usadas = float(df.loc[selecionados, "horas"].sum())

    return {
        "metodo": f"Guloso ({criterion})",
        "projetos_selecionados": selecionados,
        "n_selecionados": len(selecionados),
        "lucro_total": lucro_total,
        "custo_usado": custo_usado,
        "horas_usadas": horas_usadas,
        "tempo_execucao": tempo,
        "status": "Heurístico",
        "gap": None,  # heurística não tem garantia de gap
    }
