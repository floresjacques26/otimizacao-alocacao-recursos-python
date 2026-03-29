# -*- coding: utf-8 -*-
"""
Testes unitários — Solver Exato (PLI via PuLP).

O solver exato é a referência de qualidade do projeto.
Verificamos:
    1. A solução é factível (respeita todas as restrições)
    2. O solver retorna status "Optimal"
    3. A solução é de fato ótima em instância pequena conhecida
    4. Os campos do resultado são corretos e consistentes
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from data.data_generator import generate_projects, generate_constraints
from solvers.exact_solver import solve_exact


@pytest.fixture
def instancia_padrao():
    df = generate_projects(15, seed=42)
    constraints = generate_constraints(df, utilizacao=0.55)
    return df, constraints


class TestExactFeasibility:
    """A solução exata deve sempre ser factível."""

    def test_orcamento_respeitado(self, instancia_padrao):
        df, c = instancia_padrao
        res = solve_exact(df, c)
        assert res["custo_usado"] <= c["orcamento_maximo"] + 1e-6

    def test_horas_respeitadas(self, instancia_padrao):
        df, c = instancia_padrao
        res = solve_exact(df, c)
        assert res["horas_usadas"] <= c["horas_maximas"] + 1e-6

    def test_status_optimal(self, instancia_padrao):
        df, c = instancia_padrao
        res = solve_exact(df, c)
        assert res["status"] == "Optimal", f"Status inesperado: {res['status']}"

    def test_gap_zero(self, instancia_padrao):
        df, c = instancia_padrao
        res = solve_exact(df, c)
        assert res["gap"] == 0.0


class TestExactOptimality:
    """Verifica otimalidade em instâncias onde a resposta é conhecida."""

    def test_instancia_trivial_tudo_cabe(self):
        """Se tudo cabe no orçamento, todos os projetos devem ser selecionados."""
        df = generate_projects(5, seed=1)
        constraints = {
            "orcamento_maximo": int(df["custo"].sum()),
            "horas_maximas":    int(df["horas"].sum()),
        }
        res = solve_exact(df, constraints)
        assert res["n_selecionados"] == 5
        assert abs(res["lucro_total"] - df["lucro"].sum()) < 1e-6

    def test_instancia_trivial_nada_cabe(self):
        """Se orçamento é 0, nenhum projeto deve ser selecionado."""
        df = generate_projects(5, seed=1)
        constraints = {"orcamento_maximo": 0, "horas_maximas": 0}
        res = solve_exact(df, constraints)
        assert res["n_selecionados"] == 0
        assert res["lucro_total"] == 0.0

    def test_melhor_que_greedy(self):
        """PLI deve ter lucro >= Greedy (PLI é o upper bound)."""
        from solvers.greedy_solver import solve_greedy
        df = generate_projects(20, seed=77)
        c  = generate_constraints(df, utilizacao=0.5)
        res_exato  = solve_exact(df, c)
        res_greedy = solve_greedy(df, c)
        assert res_exato["lucro_total"] >= res_greedy["lucro_total"] - 1e-6

    def test_instancia_conhecida(self):
        """
        Instância pequena com solução conhecida manualmente.
        3 projetos: A(custo=5, horas=5, lucro=10), B(custo=4, horas=4, lucro=8), C(custo=3, horas=3, lucro=6)
        Orçamento=9, Horas=9 → ótimo = A+C (lucro=16) ou B+C (lucro=14) → A+C
        """
        import pandas as pd
        df = pd.DataFrame({
            "projeto_id": [1, 2, 3],
            "nome":       ["A", "B", "C"],
            "custo":      [5,   4,   3],
            "horas":      [5,   4,   3],
            "lucro":      [10,  8,   6],
            "eficiencia": [1.0, 1.0, 1.0],
        })
        constraints = {"orcamento_maximo": 9, "horas_maximas": 9}
        res = solve_exact(df, constraints)
        # A+B: custo=9, horas=9, lucro=18 → cabe exatamente e é o ótimo
        # A+C: custo=8, horas=8, lucro=16; B+C: custo=7, horas=7, lucro=14
        assert res["lucro_total"] == 18.0, f"Esperado 18, obtido {res['lucro_total']}"


class TestExactConsistency:
    """Consistência interna dos valores retornados."""

    def test_lucro_consistente_com_projetos(self, instancia_padrao):
        df, c = instancia_padrao
        res = solve_exact(df, c)
        lucro_calc = sum(df["lucro"].iloc[i] for i in res["projetos_selecionados"])
        assert abs(res["lucro_total"] - lucro_calc) < 1e-6

    def test_n_selecionados_consistente(self, instancia_padrao):
        df, c = instancia_padrao
        res = solve_exact(df, c)
        assert res["n_selecionados"] == len(res["projetos_selecionados"])

    def test_tempo_execucao_positivo(self, instancia_padrao):
        df, c = instancia_padrao
        res = solve_exact(df, c)
        assert res["tempo_execucao"] > 0

    def test_campos_obrigatorios(self, instancia_padrao):
        df, c = instancia_padrao
        res = solve_exact(df, c)
        campos = ["metodo", "projetos_selecionados", "n_selecionados",
                  "lucro_total", "custo_usado", "horas_usadas",
                  "tempo_execucao", "status", "gap"]
        for campo in campos:
            assert campo in res, f"Campo ausente: '{campo}'"
