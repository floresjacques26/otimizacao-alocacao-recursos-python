# -*- coding: utf-8 -*-
"""
Dashboard Streamlit — Otimização de Alocação de Recursos em Projetos
Rodar com:  streamlit run app.py
"""

import os
import sys
import time

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from data.data_generator import generate_constraints, generate_projects, get_scenarios
from solvers.exact_solver import solve_exact
from solvers.genetic_solver import solve_genetic
from solvers.greedy_solver import solve_greedy

# ─────────────────────────────────────────────────────────────────────────────
# Configuração da página
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Otimização de Recursos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS customizado
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* Fundo e tipografia geral */
    .main { background-color: #F8F9FC; }

    /* Cabeçalho hero */
    .hero-box {
        background: linear-gradient(135deg, #1565C0 0%, #0D47A1 60%, #283593 100%);
        border-radius: 14px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 1.5rem;
        color: white;
    }
    .hero-box h1 { font-size: 2rem; font-weight: 800; margin: 0 0 .4rem 0; color: white; }
    .hero-box p  { font-size: 1.05rem; margin: 0; opacity: 0.88; color: white; }

    /* Cards de métrica customizados */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        border-left: 5px solid #1565C0;
        margin-bottom: 0.5rem;
    }
    .metric-card.greedy  { border-left-color: #E65100; }
    .metric-card.genetic { border-left-color: #2E7D32; }
    .metric-label { font-size: 0.82rem; color: #666; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }
    .metric-value { font-size: 1.9rem; font-weight: 800; color: #1a1a2e; line-height: 1.1; }
    .metric-sub   { font-size: 0.85rem; color: #888; margin-top: 2px; }
    .gap-badge    { display:inline-block; background:#E8F5E9; color:#2E7D32; border-radius:20px;
                    padding:2px 10px; font-size:.8rem; font-weight:700; margin-top:4px; }
    .gap-badge.warn { background:#FFF3E0; color:#E65100; }

    /* Caixa de formulação matemática */
    .math-box {
        background: #EEF2FF;
        border-radius: 10px;
        padding: 1.2rem 1.6rem;
        border-left: 4px solid #3949AB;
        font-family: monospace;
        font-size: 0.95rem;
        line-height: 1.8;
    }

    /* Tags de status */
    .badge-otimo   { background:#E3F2FD; color:#1565C0; border-radius:20px; padding:3px 12px; font-size:.8rem; font-weight:700; }
    .badge-heur    { background:#FFF8E1; color:#F57F17; border-radius:20px; padding:3px 12px; font-size:.8rem; font-weight:700; }

    /* Esconder menu Streamlit padrão */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }

    /* Tabs mais bonitas */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constantes visuais
# ─────────────────────────────────────────────────────────────────────────────

COR_EXATO   = "#1565C0"
COR_GREEDY  = "#E65100"
COR_GENETIC = "#2E7D32"

NOMES_METODOS = {
    "exato":   "Exato (PLI)",
    "greedy":  "Guloso (Greedy)",
    "genetic": "Genético (AG)",
}

# ─────────────────────────────────────────────────────────────────────────────
# Session state — preserva resultados entre reruns do Streamlit
# ─────────────────────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "resultados": None,
        "df": None,
        "constraints": None,
        "cenario_nome": None,
        "executado": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — navegação e parâmetros
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📊 Painel de Controle")
    st.markdown("---")

    # Cenario
    st.markdown("### Cenário")
    cenario_opcao = st.radio(
        "Tamanho da instância:",
        options=["Pequeno (10 projetos)", "Médio (30 projetos)", "Grande (100 projetos)"],
        index=0,
    )
    cenario_map = {
        "Pequeno (10 projetos)":    ("pequeno",  10),
        "Médio (30 projetos)":      ("medio",    30),
        "Grande (100 projetos)":    ("grande",  100),
    }
    cenario_key, n_projetos = cenario_map[cenario_opcao]

    st.markdown("---")

    # Parametros avancados
    with st.expander("⚙️ Parâmetros Avançados", expanded=False):
        seed_val = st.number_input("Seed (reprodutibilidade)", min_value=0, max_value=9999, value=42)
        utilizacao = st.slider(
            "Utilização dos recursos (%)",
            min_value=30, max_value=80, value=55, step=5,
            help="Define o orçamento e horas disponíveis como % do total dos projetos.",
        )
        st.markdown("**Algoritmo Genético:**")
        pop_size = st.slider("População", 20, 200, 100 if n_projetos > 15 else 60, step=10)
        n_gens   = st.slider("Gerações",  50, 500, 300 if n_projetos > 15 else 150, step=50)
        mut_rate = st.slider("Taxa de mutação (%)", 1, 10, 2) / 100

    st.markdown("---")

    # Botao de execucao
    run_btn = st.button(
        "▶  Executar Simulação",
        use_container_width=True,
        type="primary",
    )

    st.markdown("---")
    st.markdown(
        """
        <small>
        <b>Projeto:</b> Otimização de Alocação de Recursos<br>
        <b>Técnicas:</b> PLI · Greedy · Algoritmo Genético<br>
        <b>Lib:</b> PuLP · NumPy · Matplotlib
        </small>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Lógica de execução dos solvers
# ─────────────────────────────────────────────────────────────────────────────

if run_btn:
    st.session_state.executado = False  # reset para mostrar progresso

    progress_bar = st.progress(0, text="Gerando dados...")
    status_area  = st.empty()

    df          = generate_projects(n_projetos, seed=int(seed_val))
    constraints = generate_constraints(df, utilizacao=utilizacao / 100)

    progress_bar.progress(15, text="[1/3] Resolvendo com PLI (PuLP/CBC)...")
    status_area.info("Solver exato em execução — Branch & Bound...")
    res_exato = solve_exact(df, constraints)

    progress_bar.progress(40, text="[2/3] Resolvendo com Greedy...")
    status_area.info("Heurística gulosa em execução...")
    res_greedy = solve_greedy(df, constraints, criterion="eficiencia")

    progress_bar.progress(60, text=f"[3/3] Algoritmo Genético (pop={pop_size}, ger={n_gens})...")
    status_area.info(f"Algoritmo Genético em execução — {n_gens} gerações, população {pop_size}...")
    res_gen = solve_genetic(
        df, constraints,
        pop_size=pop_size, n_generations=n_gens,
        mutation_rate=mut_rate, seed=int(seed_val),
    )

    progress_bar.progress(100, text="Concluído!")
    status_area.empty()
    progress_bar.empty()

    st.session_state.df          = df
    st.session_state.constraints = constraints
    st.session_state.resultados  = {
        "exato":   res_exato,
        "greedy":  res_greedy,
        "genetic": res_gen,
    }
    st.session_state.cenario_nome = cenario_opcao
    st.session_state.executado    = True
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Hero header (sempre visível)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="hero-box">
      <h1>📊 Otimização de Alocação de Recursos</h1>
      <p>Comparação entre Programação Linear Inteira (exato), Heurística Gulosa e
         Algoritmo Genético para o Problema da Mochila Multidimensional.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Estado inicial — sem execução
# ─────────────────────────────────────────────────────────────────────────────

if not st.session_state.executado:
    st.info("👈  Selecione um cenário e clique em **Executar Simulação** na barra lateral.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🎯 Problema")
        st.markdown(
            "Selecionar projetos que **maximizem o lucro total** respeitando "
            "limites de **orçamento** e **horas** da equipe."
        )
    with col2:
        st.markdown("### ⚙️ Métodos")
        st.markdown(
            "**PLI** garante otimalidade · "
            "**Greedy** é mais rápido · "
            "**AG** balanceia qualidade e velocidade."
        )
    with col3:
        st.markdown("### 📈 Métricas")
        st.markdown(
            "Lucro total, tempo de execução, gap percentual vs ótimo, "
            "utilização de recursos e convergência do AG."
        )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Dados da última execução
# ─────────────────────────────────────────────────────────────────────────────

df          = st.session_state.df
constraints = st.session_state.constraints
resultados  = st.session_state.resultados
res_exato   = resultados["exato"]
res_greedy  = resultados["greedy"]
res_gen     = resultados["genetic"]

lucro_otimo = res_exato["lucro_total"]

def _gap(lucro):
    if lucro_otimo == 0:
        return 0.0
    return ((lucro_otimo - lucro) / lucro_otimo) * 100

gap_greedy  = _gap(res_greedy["lucro_total"])
gap_genetic = _gap(res_gen["lucro_total"])

# ─────────────────────────────────────────────────────────────────────────────
# Tabs principais
# ─────────────────────────────────────────────────────────────────────────────

tab_prob, tab_res, tab_graf, tab_proj, tab_conc = st.tabs([
    "📚 Problema",
    "📊 Resultados",
    "📈 Gráficos",
    "📋 Projetos",
    "🏁 Conclusões",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — PROBLEMA
# ═════════════════════════════════════════════════════════════════════════════

with tab_prob:
    col_desc, col_form = st.columns([1, 1], gap="large")

    with col_desc:
        st.markdown("## O Problema")
        st.markdown(
            """
            Uma empresa precisa decidir **quais projetos executar** dado que possui:

            - 💰 Um **orçamento limitado** (em R\\$)
            - 🕐 Uma equipe com **horas disponiveis** limitadas

            Cada projeto candidato tem um custo, horas necessárias e lucro esperado.
            O objetivo é **maximizar o lucro total** sem exceder nenhuma restrição.
            """
        )

        st.markdown("#### Por que é difícil?")
        st.markdown(
            """
            Este problema é um caso especial da **Mochila Multidimensional 0-1**
            (_Multi-Dimensional Knapsack_), classificado como **NP-difícil**.

            Para N projetos existem **2ᴺ subconjuntos** possiveis:
            """
        )
        complexidade_df = pd.DataFrame({
            "Projetos (N)": [10, 30, 100],
            "Combinações possíveis": ["1.024", "1.073.741.824", "1,27 × 10³⁰"],
            "Força bruta": ["Trivial", "Demora horas", "Impossível"],
        })
        st.dataframe(complexidade_df, use_container_width=True, hide_index=True)

    with col_form:
        st.markdown("## Formulação Matemática")
        st.markdown(
            """
            **Variáveis de decisão:**

            xᵢ ∈ {0, 1} → 1 se o projeto i for selecionado

            **Função objetivo:**
            """
        )
        st.latex(r"\max \quad Z = \sum_{i=1}^{n} \text{lucro}_i \cdot x_i")
        st.markdown("**Restrições:**")
        st.latex(
            r"""
            \sum_{i=1}^{n} \text{custo}_i \cdot x_i \leq B \quad \text{(orçamento)}
            """
        )
        st.latex(
            r"""
            \sum_{i=1}^{n} \text{horas}_i \cdot x_i \leq H \quad \text{(horas)}
            """
        )
        st.latex(r"x_i \in \{0, 1\} \quad \forall i")

        st.markdown("---")
        st.markdown("#### Abordagens implementadas")
        abordagens = pd.DataFrame({
            "Método":["PLI (Branch & Bound)", "Greedy (Guloso)", "Algoritmo Genético"],
            "Garantia": ["Ótimo global", "Sem garantia", "Sem garantia"],
            "Complexidade": ["NP-difícil", "O(n log n)", "O(G × P × n)"],
            "Velocidade": ["Média", "Muito rápida", "Configurável"],
        })
        st.dataframe(abordagens, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(f"**Cenário atual:** {st.session_state.cenario_nome}  |  "
                f"**Projetos:** {len(df)}  |  "
                f"**Orçamento máx.:** R$ {constraints['orcamento_maximo']:,}  |  "
                f"**Horas max:** {constraints['horas_maximas']:,}h")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — RESULTADOS
# ═════════════════════════════════════════════════════════════════════════════

with tab_res:
    st.markdown(f"### Resultados — {st.session_state.cenario_nome}")
    st.markdown(
        f"Orçamento disponível: **R$ {constraints['orcamento_maximo']:,}** · "
        f"Horas disponíveis: **{constraints['horas_maximas']:,}h**"
    )
    st.markdown("---")

    # ── Metricas visuais com st.metric ────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### 🔵 Exato (PLI)")
        st.metric(
            label="Lucro Total",
            value=f"R$ {res_exato['lucro_total']:,.0f}",
            delta="Solução ótima garantida",
            delta_color="normal",
        )
        st.metric("Projetos selecionados", res_exato["n_selecionados"])
        st.metric(
            "Tempo de execução",
            f"{res_exato['tempo_execucao']*1000:.1f} ms",
        )
        st.markdown('<span class="badge-otimo">Gap = 0% — ÓTIMO</span>', unsafe_allow_html=True)

    with c2:
        st.markdown("#### 🟠 Guloso (Greedy)")
        st.metric(
            label="Lucro Total",
            value=f"R$ {res_greedy['lucro_total']:,.0f}",
            delta=f"{-gap_greedy:.2f}% vs ótimo",
            delta_color="inverse",
        )
        st.metric("Projetos selecionados", res_greedy["n_selecionados"])
        st.metric(
            "Tempo de execução",
            f"{res_greedy['tempo_execucao']*1000:.3f} ms",
        )
        badge_class = "warn" if gap_greedy > 0 else ""
        st.markdown(
            f'<span class="badge-heur">Gap = {gap_greedy:.2f}%</span>',
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown("#### 🟢 Genético (AG)")
        st.metric(
            label="Lucro Total",
            value=f"R$ {res_gen['lucro_total']:,.0f}",
            delta=f"{-gap_genetic:.2f}% vs ótimo",
            delta_color="inverse",
        )
        st.metric("Projetos selecionados", res_gen["n_selecionados"])
        st.metric(
            "Tempo de execução",
            f"{res_gen['tempo_execucao']:.2f} s",
        )
        st.markdown(
            f'<span class="badge-heur">Gap = {gap_genetic:.2f}%</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Tabela comparativa ────────────────────────────────────────────────
    st.markdown("#### Tabela Comparativa")
    tabela = pd.DataFrame([
        {
            "Método":"Exato (PLI - PuLP/CBC)",
            "Lucro Total (R$)": f"R$ {res_exato['lucro_total']:,.0f}",
            "Gap vs Ótimo":     "0.00% (ótimo)",
            "Projetos":         res_exato["n_selecionados"],
            "Orçamento Usado":  f"R$ {res_exato['custo_usado']:,.0f} ({res_exato['custo_usado']/constraints['orcamento_maximo']*100:.1f}%)",
            "Horas Usadas":     f"{res_exato['horas_usadas']:.0f}h ({res_exato['horas_usadas']/constraints['horas_maximas']*100:.1f}%)",
            "Tempo":            f"{res_exato['tempo_execucao']*1000:.1f} ms",
            "Garantia":         "Ótimo global",
        },
        {
            "Método":"Guloso (Greedy / eficiência)",
            "Lucro Total (R$)": f"R$ {res_greedy['lucro_total']:,.0f}",
            "Gap vs Ótimo":     f"{gap_greedy:.2f}%",
            "Projetos":         res_greedy["n_selecionados"],
            "Orçamento Usado":  f"R$ {res_greedy['custo_usado']:,.0f} ({res_greedy['custo_usado']/constraints['orcamento_maximo']*100:.1f}%)",
            "Horas Usadas":     f"{res_greedy['horas_usadas']:.0f}h ({res_greedy['horas_usadas']/constraints['horas_maximas']*100:.1f}%)",
            "Tempo":            f"{res_greedy['tempo_execucao']*1000:.3f} ms",
            "Garantia":         "Nenhuma",
        },
        {
            "Método":"Genético (AG)",
            "Lucro Total (R$)": f"R$ {res_gen['lucro_total']:,.0f}",
            "Gap vs Ótimo":     f"{gap_genetic:.2f}%",
            "Projetos":         res_gen["n_selecionados"],
            "Orçamento Usado":  f"R$ {res_gen['custo_usado']:,.0f} ({res_gen['custo_usado']/constraints['orcamento_maximo']*100:.1f}%)",
            "Horas Usadas":     f"{res_gen['horas_usadas']:.0f}h ({res_gen['horas_usadas']/constraints['horas_maximas']*100:.1f}%)",
            "Tempo":            f"{res_gen['tempo_execucao']:.3f} s",
            "Garantia":         "Nenhuma (meta-heurística)",
        },
    ])
    st.dataframe(tabela, use_container_width=True, hide_index=True)

    # ── Utilizacao de recursos (barras horizontais inline) ────────────────
    st.markdown("---")
    st.markdown("#### Utilização de Recursos")
    col_orc, col_hrs = st.columns(2)

    def _barra_recurso(titulo, campo, limite, unidade, col):
        with col:
            st.markdown(f"**{titulo}** (limite: {limite:,} {unidade})")
            for res, nome, cor in [
                (res_exato,  "Exato",    COR_EXATO),
                (res_greedy, "Greedy",   COR_GREEDY),
                (res_gen,    "Genético", COR_GENETIC),
            ]:
                pct = min(res[campo] / limite, 1.0)
                pct_label = f"{pct*100:.1f}%"
                st.markdown(
                    f"<div style='margin-bottom:6px'>"
                    f"<span style='font-size:.85rem;color:#555;font-weight:600'>{nome}</span> "
                    f"<span style='float:right;font-size:.85rem'>{pct_label}</span></div>",
                    unsafe_allow_html=True,
                )
                st.progress(pct)

    _barra_recurso("Orçamento (R$)", "custo_usado",  constraints["orcamento_maximo"], "R$", col_orc)
    _barra_recurso("Horas",          "horas_usadas", constraints["horas_maximas"],    "h",  col_hrs)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — GRAFICOS
# ═════════════════════════════════════════════════════════════════════════════

with tab_graf:
    st.markdown("### Visualizações")
    sel_grafico = st.selectbox(
        "Escolha o gráfico:",
        [
            "Comparação de Métodos (Lucro, Projetos, Tempo)",
            "Convergência do Algoritmo Genético",
            "Heatmap — Projetos Selecionados",
            "Distribuição dos Projetos Gerados",
        ],
    )

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110})

    # ── 1. Comparacao de metodos ──────────────────────────────────────────
    if sel_grafico == "Comparação de Métodos (Lucro, Projetos, Tempo)":
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        fig.suptitle(f"Comparação de Métodos — {st.session_state.cenario_nome}", fontweight="bold", fontsize=13)

        nomes  = ["Exato", "Greedy", "Genético"]
        lucros = [res_exato["lucro_total"], res_greedy["lucro_total"], res_gen["lucro_total"]]
        projs  = [res_exato["n_selecionados"], res_greedy["n_selecionados"], res_gen["n_selecionados"]]
        tempos = [res_exato["tempo_execucao"]*1000, res_greedy["tempo_execucao"]*1000, res_gen["tempo_execucao"]*1000]
        cores  = [COR_EXATO, COR_GREEDY, COR_GENETIC]

        for ax, vals, titulo, fmt in zip(
            axes,
            [lucros, projs, tempos],
            ["Lucro Total (R$)", "Projetos Selecionados", "Tempo (ms — log)"],
            ["R$ {:,.0f}", "{:.0f}", "{:.1f}ms"],
        ):
            bars = ax.bar(nomes, vals, color=cores, edgecolor="white", linewidth=1.5, width=0.5)
            ax.set_title(titulo, fontweight="bold", fontsize=10)
            if titulo == "Tempo (ms — log)":
                ax.set_yscale("log")
            ax.tick_params(axis="x", labelsize=9)
            ax.grid(axis="y", alpha=0.3)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                        fmt.format(v), ha="center", va="bottom", fontsize=8, fontweight="bold")
            # Anotar gap nos metodos heuristicos
            if titulo == "Lucro Total (R$)" and lucro_otimo > 0:
                for bar, v, nome in zip(bars, vals, nomes):
                    if nome != "Exato":
                        g = ((lucro_otimo - v) / lucro_otimo) * 100
                        if g > 0:
                            ax.text(bar.get_x()+bar.get_width()/2,
                                    bar.get_height()*0.5,
                                    f"gap\n{g:.1f}%",
                                    ha="center", va="center",
                                    fontsize=7.5, color="white", fontweight="bold")

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── 2. Convergencia do AG ─────────────────────────────────────────────
    elif sel_grafico == "Convergência do Algoritmo Genético":
        hist = res_gen.get("historico_convergencia", [])
        params = res_gen.get("parametros", {})
        fig, ax = plt.subplots(figsize=(11, 4.5))

        geracoes = range(1, len(hist)+1)
        ax.plot(geracoes, hist, color=COR_GENETIC, linewidth=2, label="Melhor lucro (AG)", zorder=3)
        ax.fill_between(geracoes, hist, alpha=0.12, color=COR_GENETIC)

        # Linha de convergencia (99% do valor final)
        val_final = hist[-1] if hist else 0
        gen_conv = next((i+1 for i, v in enumerate(hist) if v >= val_final * 0.99), len(hist))
        ax.axvline(gen_conv, color="orange", linestyle="--", alpha=0.85,
                   label=f"Convergência aprox. (geração {gen_conv})")

        # Linha do otimo
        ax.axhline(lucro_otimo, color=COR_EXATO, linestyle=":", linewidth=1.8,
                   label=f"Ótimo PLI: R$ {lucro_otimo:,.0f}")

        ax.set_xlabel("Geração", fontsize=11)
        ax.set_ylabel("Melhor Lucro (R$)", fontsize=11)
        ax.set_title("Convergência do Algoritmo Genético", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        col_i1, col_i2, col_i3 = st.columns(3)
        col_i1.metric("Geracoes executadas", params.get("n_generations", "—"))
        col_i2.metric("Tamanho da população", params.get("pop_size", "—"))
        col_i3.metric("Geracao de convergencia", gen_conv)

    # ── 3. Heatmap projetos selecionados ──────────────────────────────────
    elif sel_grafico == "Heatmap — Projetos Selecionados":
        import matplotlib.colors as mcolors

        n = len(df)
        metodos_list = [
            ("Exato",    res_exato,  COR_EXATO),
            ("Greedy",   res_greedy, COR_GREEDY),
            ("Genético", res_gen,    COR_GENETIC),
        ]

        matriz = np.zeros((3, n))
        for i, (_, res, _) in enumerate(metodos_list):
            for idx in res["projetos_selecionados"]:
                if 0 <= idx < n:
                    matriz[i, idx] = 1

        fig, ax = plt.subplots(figsize=(max(10, n*0.28), 3.2))
        cmap = mcolors.ListedColormap(["#ECEFF1", "#1565C0"])

        import seaborn as sns
        sns.heatmap(
            matriz, ax=ax, cmap=cmap, cbar=False,
            linewidths=0.4, linecolor="white",
            yticklabels=[m[0] for m in metodos_list],
            xticklabels=[f"P{i+1}" for i in range(n)],
        )
        ax.set_title("Projetos Selecionados por Método  (azul = selecionado)", fontweight="bold")
        ax.tick_params(axis="x", labelsize=7, rotation=45)
        ax.tick_params(axis="y", labelsize=10)

        patch_s = mpatches.Patch(color="#1565C0",  label="Selecionado")
        patch_n = mpatches.Patch(color="#ECEFF1",  label="Não selecionado")
        ax.legend(handles=[patch_s, patch_n], loc="lower right",
                  bbox_to_anchor=(1.0, -0.55), ncol=2, fontsize=9)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Projetos em comum
        sel_exato   = set(res_exato["projetos_selecionados"])
        sel_greedy  = set(res_greedy["projetos_selecionados"])
        sel_gen     = set(res_gen["projetos_selecionados"])
        consenso    = sel_exato & sel_greedy & sel_gen

        ci1, ci2, ci3 = st.columns(3)
        ci1.metric("Projetos em comum (todos os 3)", len(consenso))
        ci2.metric("Exato ∩ Greedy", len(sel_exato & sel_greedy))
        ci3.metric("Exato ∩ Genético", len(sel_exato & sel_gen))

    # ── 4. Distribuicao dos projetos ──────────────────────────────────────
    elif sel_grafico == "Distribuição dos Projetos Gerados":
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        fig.suptitle("Distribuição dos Projetos Simulados", fontsize=13, fontweight="bold")

        axes[0,0].hist(df["custo"], bins=12, color=COR_EXATO, alpha=0.75, edgecolor="white")
        axes[0,0].set_title("Distribuição de Custos (R$)")
        axes[0,0].set_xlabel("Custo")
        axes[0,0].set_ylabel("Frequência")
        axes[0,0].grid(axis="y", alpha=0.3)

        axes[0,1].hist(df["horas"], bins=12, color=COR_GREEDY, alpha=0.75, edgecolor="white")
        axes[0,1].set_title("Distribuição de Horas")
        axes[0,1].set_xlabel("Horas")
        axes[0,1].set_ylabel("Frequência")
        axes[0,1].grid(axis="y", alpha=0.3)

        sc = axes[1,0].scatter(df["custo"], df["lucro"], c=df["eficiencia"],
                                cmap="RdYlGn", s=60, alpha=0.75, edgecolors="white", linewidths=0.4)
        axes[1,0].set_title("Custo x Lucro (cor = eficiência)")
        axes[1,0].set_xlabel("Custo (R$)")
        axes[1,0].set_ylabel("Lucro (R$)")
        plt.colorbar(sc, ax=axes[1,0], label="Eficiência")

        top20 = df.nlargest(min(20, len(df)), "eficiencia")
        cores_rank = plt.cm.RdYlGn(
            (top20["eficiencia"] - top20["eficiencia"].min())
            / (top20["eficiencia"].max() - top20["eficiencia"].min() + 1e-9)
        )
        axes[1,1].barh(top20["nome"], top20["eficiencia"], color=cores_rank)
        axes[1,1].invert_yaxis()
        axes[1,1].set_title(f"Top {len(top20)} por Eficiência")
        axes[1,1].set_xlabel("Eficiência (lucro/recurso)")

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — PROJETOS
# ═════════════════════════════════════════════════════════════════════════════

with tab_proj:
    st.markdown("### Tabela de Projetos")
    st.markdown(
        f"**{len(df)} projetos gerados** · "
        f"Orçamento total dos projetos: R$ {df['custo'].sum():,} · "
        f"Horas totais: {df['horas'].sum():,}h"
    )

    # Filtros
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        custo_range = st.slider(
            "Filtrar por custo (R$)",
            int(df["custo"].min()), int(df["custo"].max()),
            (int(df["custo"].min()), int(df["custo"].max())),
        )
    with col_f2:
        lucro_range = st.slider(
            "Filtrar por lucro (R$)",
            int(df["lucro"].min()), int(df["lucro"].max()),
            (int(df["lucro"].min()), int(df["lucro"].max())),
        )

    df_filtrado = df[
        (df["custo"].between(*custo_range)) &
        (df["lucro"].between(*lucro_range))
    ].copy()

    # Adicionar coluna indicando quais metodos selecionaram cada projeto
    sel_e = set(res_exato["projetos_selecionados"])
    sel_g = set(res_greedy["projetos_selecionados"])
    sel_n = set(res_gen["projetos_selecionados"])

    def _marcadores(idx):
        tags = []
        if idx in sel_e: tags.append("PLI")
        if idx in sel_g: tags.append("Greedy")
        if idx in sel_n: tags.append("AG")
        return ", ".join(tags) if tags else "—"

    df_filtrado["Selecionado por"] = df_filtrado.index.map(_marcadores)
    df_filtrado["eficiencia"] = df_filtrado["eficiencia"].round(3)

    df_display = df_filtrado[["nome", "custo", "horas", "lucro", "eficiencia", "Selecionado por"]].copy()
    df_display.columns = ["Projeto", "Custo (R$)", "Horas", "Lucro (R$)", "Eficiência", "Selecionado por"]

    st.dataframe(
        df_display.sort_values("Eficiência", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"Mostrando {len(df_filtrado)} de {len(df)} projetos.")

    st.markdown("---")
    st.markdown("#### Projetos por Método")

    tab_e, tab_g, tab_n = st.tabs(["🔵 Exato (PLI)", "🟠 Greedy", "🟢 Genético (AG)"])

    def _tabela_selecionados(res, tab):
        with tab:
            idxs = res["projetos_selecionados"]
            if not idxs:
                st.warning("Nenhum projeto selecionado.")
                return
            df_sel = df.loc[idxs, ["nome","custo","horas","lucro","eficiencia"]].copy()
            df_sel["eficiencia"] = df_sel["eficiencia"].round(3)
            df_sel.columns = ["Projeto", "Custo (R$)", "Horas", "Lucro (R$)", "Eficiência"]
            st.dataframe(df_sel.sort_values("Eficiência", ascending=False),
                         use_container_width=True, hide_index=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Total projetos",      res["n_selecionados"])
            c2.metric("Lucro total",         f"R$ {res['lucro_total']:,.0f}")
            c3.metric("Orçamento consumido", f"R$ {res['custo_usado']:,.0f}")

    _tabela_selecionados(res_exato,  tab_e)
    _tabela_selecionados(res_greedy, tab_g)
    _tabela_selecionados(res_gen,    tab_n)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — CONCLUSOES
# ═════════════════════════════════════════════════════════════════════════════

with tab_conc:
    st.markdown("## Análise Comparativa e Conclusões")

    # ── Veredito automatico ───────────────────────────────────────────────
    velocidade_greedy  = res_greedy["tempo_execucao"]
    velocidade_exato   = res_exato["tempo_execucao"]
    speedup_greedy     = velocidade_exato / velocidade_greedy if velocidade_greedy > 0 else 0

    if gap_genetic < 0.5:
        veredito_ag = "excelente"
        cor_veredito = "normal"
    elif gap_genetic < 5:
        veredito_ag = "bom"
        cor_veredito = "normal"
    else:
        veredito_ag = "moderado"
        cor_veredito = "off"

    st.success(
        f"**Resultado do cenário '{cenario_opcao}':**  "
        f"O Greedy foi **{speedup_greedy:,.0f}x mais rápido** que o PLI e teve gap de "
        f"**{gap_greedy:.2f}%**.  "
        f"O Algoritmo Genético obteve desempenho **{veredito_ag}** com gap de "
        f"**{gap_genetic:.2f}%**."
    )

    st.markdown("---")

    # ── Analise qualitativa ───────────────────────────────────────────────
    col_q1, col_q2 = st.columns(2)

    with col_q1:
        st.markdown("#### Quando usar cada método?")
        st.markdown("""
| Situação | Recomendação |
|----------|-------------|
| Precisa da solução **perfeita** | PLI (Branch & Bound) |
| Precisa de **velocidade máxima** | Greedy |
| Problema **grande** + boa qualidade | Algoritmo Genético |
| Instância pequena (< 50 projetos) | PLI sempre |
| Decisão em **tempo real** | Greedy |
        """)

    with col_q2:
        st.markdown("#### Análise de Complexidade")
        complexidade = pd.DataFrame({
            "Método":["PLI (Branch & Bound)", "Greedy", "Algoritmo Genético"],
            "Complexidade": ["NP-difícil", "O(n log n)", "O(G × P × n)"],
            "Gap Típico": ["0%", "0–15%", "0–5%"],
            "Escalabilidade": ["Limitada", "Excelente", "Boa"],
            "Determinismo": ["Sim", "Sim", "Não (aleatório)"],
        })
        st.dataframe(complexidade, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Grafico de resumo final ───────────────────────────────────────────
    st.markdown("#### Gráfico Resumo — Qualidade x Velocidade")

    fig, ax = plt.subplots(figsize=(9, 4))

    qualidades = [
        lucro_otimo / lucro_otimo * 100,
        res_greedy["lucro_total"] / lucro_otimo * 100 if lucro_otimo > 0 else 0,
        res_gen["lucro_total"] / lucro_otimo * 100 if lucro_otimo > 0 else 0,
    ]
    velocidades = [
        res_exato["tempo_execucao"],
        res_greedy["tempo_execucao"],
        res_gen["tempo_execucao"],
    ]
    labels = ["PLI (Exato)", "Greedy", "Genético (AG)"]
    cores_scatter = [COR_EXATO, COR_GREEDY, COR_GENETIC]

    for x, y, label, cor in zip(velocidades, qualidades, labels, cores_scatter):
        ax.scatter(x, y, color=cor, s=250, zorder=5, edgecolors="white", linewidths=1.5)
        ax.annotate(
            label, (x, y),
            textcoords="offset points", xytext=(10, 5),
            fontsize=10, fontweight="bold", color=cor,
        )

    ax.axhline(y=99, color="gray", linestyle="--", alpha=0.4, label="99% do ótimo")
    ax.set_xlabel("Tempo de execução (s)", fontsize=11)
    ax.set_ylabel("Qualidade da solução (% do ótimo)", fontsize=11)
    ax.set_title("Trade-off: Qualidade × Velocidade", fontsize=12, fontweight="bold")
    ax.set_xscale("log")
    ax.set_ylim(max(50, min(qualidades) - 10), 105)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")

    # ── Proximos passos ───────────────────────────────────────────────────
    with st.expander("🚀 Possíveis melhorias e próximos passos"):
        st.markdown("""
        **Algoritmos:**
        - **GRASP** (Greedy Randomized Adaptive Search) — combina aleatoriedade com busca local
        - **Simulated Annealing** — aceita soluções piores com probabilidade decrescente
        - **NSGA-II** — evolução multi-objetivo (lucro x risco, por exemplo)
        - **Branch & Price** — PLI com geração de colunas para instâncias muito grandes

        **Modelagem:**
        - Dependências entre projetos (A só é executado se B também for)
        - Múltiplas equipes com diferentes especialidades
        - Horizonte temporal com restrições de precedência
        - Incerteza no lucro (otimização robusta ou estocástica)

        **Engenharia:**
        - Exportar resultados em CSV/JSON
        - Testes unitarios com pytest
        - Profiling do Algoritmo Genético (line_profiler)
        - Deploy do dashboard no Streamlit Cloud
        """)

    # ── Rodape ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<center><small>Projeto desenvolvido para portfólio de pesquisa em otimização combinatória.</small></center>",
        unsafe_allow_html=True,
    )
