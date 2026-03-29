# -*- coding: utf-8 -*-
"""
=============================================================================
   OTIMIZACAO DE ALOCACAO DE RECURSOS EM PROJETOS
   Comparacao: PLI (Exato) x Greedy x Algoritmo Genetico
=============================================================================

  Problema:
    Dados N projetos candidatos (com custo, horas e lucro), selecionar
    quais executar para MAXIMIZAR o lucro total respeitando:
      - Orcamento maximo disponivel (R$)
      - Horas maximas da equipe

    Formulacao (Mochila Multidimensional 0-1):
      max  sum(lucro_i * x_i)
      s.t. sum(custo_i * x_i) <= B
           sum(horas_i * x_i) <= H
           x_i in {0, 1}

  Uso:  python main.py
=============================================================================
"""

import os
import sys
import time

import pandas as pd

# Adicionar src ao path para imports sem instalar o pacote
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from data.data_generator import generate_constraints, generate_projects, get_scenarios
from solvers.exact_solver import solve_exact
from solvers.genetic_solver import solve_genetic
from solvers.greedy_solver import solve_greedy
from visualization.plots import (
    plot_comparacao_cenarios,
    plot_comparacao_metodos,
    plot_convergencia_genetico,
    plot_distribuicao_projetos,
    plot_projetos_selecionados,
    plot_utilizacao_recursos,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuração
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Controla se os gráficos abrem em janela interativa (True) ou apenas são salvos
SHOW_PLOTS = False  # Mude para True se quiser ver os gráficos na tela


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de output
# ─────────────────────────────────────────────────────────────────────────────

def _cabecalho() -> None:
    print()
    print("=" * 72)
    print("  OTIMIZACAO DE ALOCACAO DE RECURSOS EM PROJETOS")
    print("  PLI (Exato)  x  Greedy  x  Algoritmo Genetico")
    print("=" * 72)


def _tabela_resultados(resultados: list, constraints: dict, titulo: str) -> None:
    """Imprime tabela comparativa formatada com gap percentual vs otimo."""
    print()
    print(f"  +{'-'*68}+")
    print(f"  |  {titulo:<66}|")
    print(f"  |  Orcamento: R$ {constraints['orcamento_maximo']:,}  |  Horas: {constraints['horas_maximas']:,}h{'':<25}|")
    print(f"  +{'-'*68}+")

    header = (
        f"  {'Metodo':<28}  {'Lucro':>10}  {'Gap':>7}  "
        f"{'Proj':>5}  {'Orc%':>6}  {'Hrs%':>6}  {'Tempo':>9}"
    )
    print(header)
    print("  " + "-" * 68)

    lucro_otimo = next(
        (r["lucro_total"] for r in resultados if "Exato" in r["metodo"]), None
    )

    for res in resultados:
        pct_orc = (res["custo_usado"] / constraints["orcamento_maximo"]) * 100
        pct_hrs = (res["horas_usadas"] / constraints["horas_maximas"]) * 100

        if lucro_otimo and "Exato" not in res["metodo"] and lucro_otimo > 0:
            gap_str = f"{((lucro_otimo - res['lucro_total']) / lucro_otimo * 100):+.1f}%"
        elif "Exato" in res["metodo"]:
            gap_str = "otimo"
        else:
            gap_str = "-"

        tempo_ms = res["tempo_execucao"] * 1000
        tempo_str = f"{tempo_ms:.1f}ms" if tempo_ms >= 0.1 else f"{tempo_ms:.3f}ms"

        print(
            f"  {res['metodo']:<28}  "
            f"R$ {res['lucro_total']:>8,.0f}  "
            f"{gap_str:>7}  "
            f"{res['n_selecionados']:>5}  "
            f"{pct_orc:>5.1f}%  "
            f"{pct_hrs:>5.1f}%  "
            f"{tempo_str:>9}"
        )

    print("  " + "-" * 68)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Execucao de um cenario
# ─────────────────────────────────────────────────────────────────────────────

def executar_cenario(nome: str, n_projetos: int, seed: int = 42) -> tuple:
    """
    Executa os tres solvers para um cenario e retorna (df, constraints, resultados).
    Os parametros do AG sao ajustados ao tamanho do problema.
    """
    print(f"\n  -- {nome} --")
    print(f"  Gerando {n_projetos} projetos (seed={seed})...")

    df = generate_projects(n_projetos, seed=seed)
    constraints = generate_constraints(df, utilizacao=0.55)

    print(
        f"  Orçamento máximo: R$ {constraints['orcamento_maximo']:,}  |  "
        f"Horas máximas: {constraints['horas_maximas']:,}h"
    )

    resultados = []

    # ── 1. Solução Exata (PLI) ────────────────────────────────────────────
    print("  [1/3] PLI (PuLP/CBC)    ...", end=" ", flush=True)
    res_exato = solve_exact(df, constraints)
    print(
        f"R$ {res_exato['lucro_total']:,.0f}  "
        f"({res_exato['tempo_execucao']*1000:.1f}ms)  "
        f"[{res_exato['status']}]"
    )
    resultados.append(res_exato)

    # ── 2. Greedy ─────────────────────────────────────────────────────────
    print("  [2/3] Greedy            ...", end=" ", flush=True)
    res_greedy = solve_greedy(df, constraints, criterion="eficiencia")
    print(
        f"R$ {res_greedy['lucro_total']:,.0f}  "
        f"({res_greedy['tempo_execucao']*1000:.3f}ms)"
    )
    resultados.append(res_greedy)

    # ── 3. Algoritmo Genético ─────────────────────────────────────────────
    # Para problemas pequenos usamos populações menores (mais rápido)
    pop   = 60  if n_projetos <= 15 else 120
    gens  = 150 if n_projetos <= 15 else 300

    print(
        f"  [3/3] Genético (pop={pop}, ger={gens}) ...",
        end=" ", flush=True,
    )
    res_gen = solve_genetic(df, constraints, pop_size=pop, n_generations=gens, seed=seed)
    print(
        f"R$ {res_gen['lucro_total']:,.0f}  "
        f"({res_gen['tempo_execucao']:.2f}s)"
    )
    resultados.append(res_gen)

    return df, constraints, resultados


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _cabecalho()

    scenarios = get_scenarios()
    resultados_cenarios: dict = {}

    # ── Executar cada cenário ─────────────────────────────────────────────
    for nome, config in scenarios.items():

        df, constraints, resultados = executar_cenario(
            nome=config["descricao"],
            n_projetos=config["n_projetos"],
            seed=42,
        )
        resultados_cenarios[nome] = resultados

        # Imprimir tabela de resultados no terminal
        _tabela_resultados(resultados, constraints, config["descricao"])

        # ── Gráficos por cenário ──────────────────────────────────────────
        print("  Gerando gráficos...")

        plot_comparacao_metodos(
            resultados, constraints,
            titulo=config["descricao"],
            save_path=os.path.join(RESULTS_DIR, f"comparacao_{nome}.png"),
            show=SHOW_PLOTS,
        )

        plot_utilizacao_recursos(
            resultados, constraints,
            titulo=config["descricao"],
            save_path=os.path.join(RESULTS_DIR, f"utilizacao_{nome}.png"),
            show=SHOW_PLOTS,
        )

        # Convergência do AG (disponível em todos os cenários)
        res_gen = next(r for r in resultados if "Genético" in r["metodo"])
        res_exato = next((r for r in resultados if "Exato" in r["metodo"]), None)
        lucro_otimo = res_exato["lucro_total"] if res_exato else None

        plot_convergencia_genetico(
            res_gen["historico_convergencia"],
            lucro_otimo=lucro_otimo,
            save_path=os.path.join(RESULTS_DIR, f"convergencia_ag_{nome}.png"),
            show=SHOW_PLOTS,
        )

        # Visualizações detalhadas apenas no cenário pequeno (mais legíveis)
        if nome == "pequeno":
            plot_distribuicao_projetos(
                df,
                save_path=os.path.join(RESULTS_DIR, "distribuicao_projetos.png"),
                show=SHOW_PLOTS,
            )
            plot_projetos_selecionados(
                df, resultados,
                save_path=os.path.join(RESULTS_DIR, "projetos_selecionados.png"),
                show=SHOW_PLOTS,
            )

    # ── Comparação cross-cenário ──────────────────────────────────────────
    print("\n  Gerando comparação entre cenários...")
    plot_comparacao_cenarios(
        resultados_cenarios,
        save_path=os.path.join(RESULTS_DIR, "comparacao_cenarios.png"),
        show=SHOW_PLOTS,
    )

    # Analise de complexidade computacional
    print()
    print("=" * 72)
    print("  COMPLEXIDADE COMPUTACIONAL")
    print("=" * 72)
    print("""
  PLI com Branch & Bound (solver CBC):
    - Complexidade: NP-dificil no pior caso
    - Na pratica: Branch & Bound + relaxacao LP pode ser muito rapido
    - GARANTE solucao globalmente otima (gap = 0%)
    - Recomendado para: instancias pequenas/medias (< 500 variaveis)

  Greedy:
    - Complexidade: O(n log n) -- dominado pela ordenacao
    - Extremamente rapido; escala bem para milhares de projetos
    - NAO garante otimalidade (pode errar em ~5-15%)
    - Recomendado para: decisoes rapidas, dados em tempo real

  Algoritmo Genetico:
    - Complexidade: O(G x P x n)  [geracoes x populacao x projetos]
    - Flexivel: melhora com mais geracoes/populacao
    - NAO garante otimalidade, mas tende a superar o Greedy
    - Recomendado para: problemas grandes onde PLI e impraticavel
    """)

    print("=" * 72)
    print(f"  Todos os graficos salvos em: {RESULTS_DIR}/")
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()
