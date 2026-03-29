# -*- coding: utf-8 -*-
"""
Módulo de visualização dos resultados.

Gráficos disponíveis:
    1. plot_comparacao_metodos     → lucro, projetos e tempo por método
    2. plot_utilizacao_recursos    → uso de orçamento e horas por método
    3. plot_convergencia_genetico  → curva de convergência do AG por geração
    4. plot_projetos_selecionados  → heatmap de quais projetos cada método escolheu
    5. plot_distribuicao_projetos  → distribuição dos dados gerados
    6. plot_comparacao_cenarios    → desempenho nos cenários pequeno/médio/grande
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Configurações globais de estilo ───────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.3,
        "figure.dpi": 110,
    }
)

# Paleta de cores consistente para cada método
CORES = {
    "Exato (PLI - PuLP/CBC)": "#1565C0",   # azul escuro — solução de referência
    "Guloso (eficiencia)":     "#E65100",   # laranja — heurística simples
    "Genético (AG)":           "#2E7D32",   # verde — heurística avançada
}

# Fallback para métodos com nomes ligeiramente diferentes
def _cor(metodo: str) -> str:
    for chave, cor in CORES.items():
        if any(k in metodo for k in ["Exato", "PLI"]):
            return "#1565C0"
        if any(k in metodo for k in ["Guloso", "greedy", "eficiencia", "lucro", "custo"]):
            return "#E65100"
        if any(k in metodo for k in ["Genético", "AG"]):
            return "#2E7D32"
    return "#7B1FA2"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Comparação de métodos
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparacao_metodos(
    resultados: list,
    constraints: dict,
    titulo: str = "",
    save_path: str = None,
    show: bool = True,
) -> None:
    """
    Painel com 3 gráficos de barras lado a lado:
        (1) Lucro total obtido por método
        (2) Número de projetos selecionados
        (3) Tempo de execução em milissegundos (escala log)

    Inclui anotação do gap percentual em relação à solução ótima.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Comparação de Métodos — {titulo}", fontsize=13, fontweight="bold", y=1.02)

    metodos = [r["metodo"] for r in resultados]
    lucros   = [r["lucro_total"] for r in resultados]
    n_proj   = [r["n_selecionados"] for r in resultados]
    tempos   = [r["tempo_execucao"] * 1000 for r in resultados]  # → ms
    cores    = [_cor(m) for m in metodos]

    # Lucro do método exato como referência para calcular o gap
    lucro_otimo = next((r["lucro_total"] for r in resultados if "Exato" in r["metodo"]), None)

    # ── Gráfico 1: Lucro total ────────────────────────────────────────────
    ax1 = axes[0]
    bars = ax1.bar(range(len(metodos)), lucros, color=cores, edgecolor="white", linewidth=1.5, width=0.5)
    ax1.set_title("Lucro Total (R$)", fontweight="bold")
    ax1.set_ylabel("Lucro (R$)")
    ax1.set_xticks(range(len(metodos)))
    ax1.set_xticklabels([m.split(" ")[0] for m in metodos], rotation=10)

    for bar, val, metodo in zip(bars, lucros, metodos):
        label = f"R$ {val:,.0f}"
        if lucro_otimo and "Exato" not in metodo and lucro_otimo > 0:
            gap = ((lucro_otimo - val) / lucro_otimo) * 100
            label += f"\n(gap {gap:.1f}%)"
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            label,
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )

    # ── Gráfico 2: Projetos selecionados ──────────────────────────────────
    ax2 = axes[1]
    bars2 = ax2.bar(range(len(metodos)), n_proj, color=cores, edgecolor="white", linewidth=1.5, width=0.5)
    ax2.set_title("Projetos Selecionados", fontweight="bold")
    ax2.set_ylabel("Quantidade")
    ax2.set_xticks(range(len(metodos)))
    ax2.set_xticklabels([m.split(" ")[0] for m in metodos], rotation=10)

    for bar, val in zip(bars2, n_proj):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            str(val),
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    # ── Gráfico 3: Tempo de execução (log) ────────────────────────────────
    ax3 = axes[2]
    bars3 = ax3.bar(range(len(metodos)), tempos, color=cores, edgecolor="white", linewidth=1.5, width=0.5)
    ax3.set_title("Tempo de Execução (ms)", fontweight="bold")
    ax3.set_ylabel("Tempo (ms) — escala log")
    ax3.set_yscale("log")
    ax3.set_xticks(range(len(metodos)))
    ax3.set_xticklabels([m.split(" ")[0] for m in metodos], rotation=10)

    for bar, val in zip(bars3, tempos):
        label = f"{val:.1f}ms" if val >= 1 else f"{val:.3f}ms"
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.3,
            label,
            ha="center", va="bottom", fontsize=8,
        )

    plt.tight_layout()
    _salvar_e_exibir(save_path, show)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Utilização de recursos
# ─────────────────────────────────────────────────────────────────────────────

def plot_utilizacao_recursos(
    resultados: list,
    constraints: dict,
    titulo: str = "",
    save_path: str = None,
    show: bool = True,
) -> None:
    """
    Barras horizontais mostrando o percentual de uso do orçamento e das horas
    para cada método. A linha vermelha tracejada marca o limite de 100%.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Utilização de Recursos — {titulo}", fontsize=13, fontweight="bold")

    for col_idx, (campo, nome) in enumerate(
        [("custo_usado", "Orçamento"), ("horas_usadas", "Horas")]
    ):
        ax = axes[col_idx]
        limite = constraints["orcamento_maximo"] if campo == "custo_usado" else constraints["horas_maximas"]
        unidade = "R$" if campo == "custo_usado" else "h"

        for i, res in enumerate(resultados):
            pct = (res[campo] / limite) * 100
            cor = _cor(res["metodo"])
            ax.barh(i, pct, color=cor, alpha=0.85, edgecolor="white", height=0.5)
            ax.text(
                pct + 0.5, i,
                f"{pct:.1f}%  ({res[campo]:,.0f} {unidade})",
                va="center", fontsize=9,
            )

        # Linha de limite
        ax.axvline(x=100, color="red", linestyle="--", linewidth=1.5, label="Limite (100%)")
        ax.set_xlim(0, 125)
        ax.set_yticks(range(len(resultados)))
        ax.set_yticklabels([r["metodo"].split(" ")[0] for r in resultados])
        ax.set_xlabel(f"Utilização do {nome} (%)")
        ax.set_title(f"Uso de {nome} (limite: {limite:,} {unidade})", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="x", alpha=0.3)
        ax.grid(axis="y", alpha=0)

    plt.tight_layout()
    _salvar_e_exibir(save_path, show)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Convergência do algoritmo genético
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergencia_genetico(
    historico: list,
    lucro_otimo: float = None,
    save_path: str = None,
    show: bool = True,
) -> None:
    """
    Curva de convergência do AG:
        - Eixo X: geração
        - Eixo Y: melhor lucro encontrado até aquela geração

    Opcional: linha horizontal indicando o ótimo global (PLI) para referência.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    geracoes = range(1, len(historico) + 1)
    cor_ag = "#2E7D32"

    ax.plot(geracoes, historico, color=cor_ag, linewidth=2, label="Melhor fitness (AG)", zorder=3)
    ax.fill_between(geracoes, historico, alpha=0.12, color=cor_ag)

    # Marcar o ponto onde o AG atinge 99% do valor final (proxy de convergência)
    val_final = historico[-1]
    gen_conv = next((i + 1 for i, v in enumerate(historico) if v >= val_final * 0.99), len(historico))
    ax.axvline(
        x=gen_conv, color="orange", linestyle="--", alpha=0.8,
        label=f"Convergência ≈ geração {gen_conv}",
    )

    # Linha do ótimo global para comparação (se fornecido)
    if lucro_otimo:
        ax.axhline(
            y=lucro_otimo, color="#1565C0", linestyle=":", linewidth=1.8,
            label=f"Ótimo PLI: R$ {lucro_otimo:,.0f}",
        )

    ax.set_xlabel("Geração", fontsize=12)
    ax.set_ylabel("Melhor Lucro (R$)", fontsize=12)
    ax.set_title("Convergência do Algoritmo Genético", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _salvar_e_exibir(save_path, show)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Heatmap de projetos selecionados
# ─────────────────────────────────────────────────────────────────────────────

def plot_projetos_selecionados(
    df: pd.DataFrame,
    resultados: list,
    save_path: str = None,
    show: bool = True,
) -> None:
    """
    Heatmap binário mostrando quais projetos cada método selecionou.
    Azul = selecionado, cinza claro = não selecionado.
    """
    n = len(df)
    n_metodos = len(resultados)

    # Construir matriz de seleção (n_metodos × n_projetos)
    matriz = np.zeros((n_metodos, n))
    for i, res in enumerate(resultados):
        for idx in res["projetos_selecionados"]:
            if 0 <= idx < n:
                matriz[i, idx] = 1

    fig_width = max(12, n * 0.35)
    fig, ax = plt.subplots(figsize=(fig_width, 3))

    sns.heatmap(
        matriz,
        ax=ax,
        cmap=["#ECEFF1", "#1565C0"],
        cbar=False,
        linewidths=0.4,
        linecolor="white",
        yticklabels=[r["metodo"].split(" ")[0] for r in resultados],
        xticklabels=[f"P{i+1}" for i in range(n)],
    )

    ax.set_title(
        "Projetos Selecionados por Metodo  (azul = selecionado  |  cinza = nao selecionado)",
        fontweight="bold",
    )
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    ax.tick_params(axis="y", labelsize=10)

    # Legenda manual
    patch_sim = mpatches.Patch(color="#1565C0", label="Selecionado")
    patch_nao = mpatches.Patch(color="#ECEFF1", label="Não selecionado")
    ax.legend(handles=[patch_sim, patch_nao], loc="lower right",
               bbox_to_anchor=(1.0, -0.35), ncol=2, fontsize=9)

    plt.tight_layout()
    _salvar_e_exibir(save_path, show)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Distribuição dos dados gerados
# ─────────────────────────────────────────────────────────────────────────────

def plot_distribuicao_projetos(
    df: pd.DataFrame,
    save_path: str = None,
    show: bool = True,
) -> None:
    """
    Painel 2×2 com:
        (TL) Histograma de custos
        (TR) Histograma de horas
        (BL) Scatter custo × lucro colorido por eficiência
        (BR) Ranking dos top-20 projetos por eficiência
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Distribuição dos Projetos Simulados", fontsize=14, fontweight="bold")

    # Histograma de custos
    axes[0, 0].hist(df["custo"], bins=12, color="#1565C0", alpha=0.75, edgecolor="white")
    axes[0, 0].set_title("Distribuição de Custos")
    axes[0, 0].set_xlabel("Custo (R$)")
    axes[0, 0].set_ylabel("Frequência")

    # Histograma de horas
    axes[0, 1].hist(df["horas"], bins=12, color="#E65100", alpha=0.75, edgecolor="white")
    axes[0, 1].set_title("Distribuição de Horas Necessárias")
    axes[0, 1].set_xlabel("Horas")
    axes[0, 1].set_ylabel("Frequência")

    # Scatter custo × lucro, cor = eficiência
    scatter = axes[1, 0].scatter(
        df["custo"], df["lucro"],
        c=df["eficiencia"], cmap="RdYlGn",
        s=70, alpha=0.75, edgecolors="white", linewidths=0.5,
    )
    axes[1, 0].set_title("Custo × Lucro  (cor = eficiência)")
    axes[1, 0].set_xlabel("Custo (R$)")
    axes[1, 0].set_ylabel("Lucro (R$)")
    plt.colorbar(scatter, ax=axes[1, 0], label="Eficiência")

    # Ranking por eficiência (top 20)
    top = df.nlargest(min(20, len(df)), "eficiencia")
    cores_rank = plt.cm.RdYlGn(
        (top["eficiencia"] - top["eficiencia"].min())
        / (top["eficiencia"].max() - top["eficiencia"].min() + 1e-9)
    )
    axes[1, 1].barh(top["nome"], top["eficiencia"], color=cores_rank)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_title("Top 20 por Eficiência (lucro/recurso)")
    axes[1, 1].set_xlabel("Eficiência")

    plt.tight_layout()
    _salvar_e_exibir(save_path, show)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Comparação entre cenários
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparacao_cenarios(
    resultados_cenarios: dict,
    save_path: str = None,
    show: bool = True,
) -> None:
    """
    Gráfico de barras agrupadas comparando os métodos nos cenários
    pequeno, médio e grande para as métricas:
        (1) Lucro total
        (2) Tempo de execução (log)
        (3) Gap percentual em relação ao ótimo
    """
    cenarios = list(resultados_cenarios.keys())
    x = np.arange(len(cenarios))

    # Coletar todos os métodos presentes
    metodos_set: list = []
    for res_list in resultados_cenarios.values():
        for r in res_list:
            if r["metodo"] not in metodos_set:
                metodos_set.append(r["metodo"])

    n_metodos = len(metodos_set)
    width = 0.7 / n_metodos

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Desempenho Comparativo por Cenário", fontsize=13, fontweight="bold")

    metricas = [
        ("lucro_total",     "Lucro Total (R$)",          False),
        ("tempo_execucao",  "Tempo de Execução (s)",      True),
    ]

    for ax_idx, (metrica, titulo_ax, log_scale) in enumerate(metricas):
        ax = axes[ax_idx]
        for i, metodo in enumerate(metodos_set):
            valores = []
            for cenario in cenarios:
                val = next(
                    (r[metrica] for r in resultados_cenarios[cenario] if r["metodo"] == metodo),
                    0,
                )
                valores.append(val)

            offset = (i - n_metodos / 2 + 0.5) * width
            ax.bar(x + offset, valores, width * 0.9, label=metodo.split(" ")[0],
                   color=_cor(metodo), alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([c.capitalize() for c in cenarios])
        ax.set_title(titulo_ax, fontweight="bold")
        ax.legend(fontsize=8)
        if log_scale:
            ax.set_yscale("log")
            ax.set_ylabel("Tempo (s) — escala log")

    # ── Gráfico 3: Gap percentual ─────────────────────────────────────────
    ax3 = axes[2]
    metodos_heur = [m for m in metodos_set if "Exato" not in m]

    for i, metodo in enumerate(metodos_heur):
        gaps = []
        for cenario in cenarios:
            res_list = resultados_cenarios[cenario]
            lucro_otimo = next(
                (r["lucro_total"] for r in res_list if "Exato" in r["metodo"]), None
            )
            lucro_heur = next(
                (r["lucro_total"] for r in res_list if r["metodo"] == metodo), 0
            )
            if lucro_otimo and lucro_otimo > 0:
                gaps.append(((lucro_otimo - lucro_heur) / lucro_otimo) * 100)
            else:
                gaps.append(0)

        offset = (i - len(metodos_heur) / 2 + 0.5) * width
        ax3.bar(x + offset, gaps, width * 0.9, label=metodo.split(" ")[0],
                color=_cor(metodo), alpha=0.85)

    ax3.set_xticks(x)
    ax3.set_xticklabels([c.capitalize() for c in cenarios])
    ax3.set_title("Gap vs Ótimo (%)", fontweight="bold")
    ax3.set_ylabel("Gap (%)")
    ax3.axhline(y=0, color="black", linewidth=0.8)
    ax3.legend(fontsize=8)

    plt.tight_layout()
    _salvar_e_exibir(save_path, show)


# ─────────────────────────────────────────────────────────────────────────────
# Utilitário interno
# ─────────────────────────────────────────────────────────────────────────────

def _salvar_e_exibir(save_path: str, show: bool) -> None:
    """Salva o gráfico em arquivo e/ou exibe na tela conforme os parâmetros."""
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"    [OK] Grafico salvo em: {save_path}")
    if show:
        plt.show()
    plt.close()
