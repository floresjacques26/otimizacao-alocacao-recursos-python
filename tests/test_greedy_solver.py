"""
Testes unitários — Solver Heurístico Guloso.

Propriedades críticas verificadas:
    1. A solução NUNCA viola o orçamento
    2. A solução NUNCA viola as horas
    3. O lucro retornado é consistente com os projetos selecionados
    4. Todos os campos obrigatórios estão presentes no resultado
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from data.data_generator import generate_projects, generate_constraints
from solvers.greedy_solver import solve_greedy


@pytest.fixture
def instancia_pequena():
    df = generate_projects(10, seed=42)
    constraints = generate_constraints(df, utilizacao=0.55)
    return df, constraints


@pytest.fixture
def instancia_grande():
    df = generate_projects(100, seed=99)
    constraints = generate_constraints(df, utilizacao=0.55)
    return df, constraints


class TestGreedyFeasibility:
    """O Greedy deve sempre retornar soluções factíveis."""

    def test_orcamento_nunca_excedido(self, instancia_pequena):
        df, c = instancia_pequena
        res = solve_greedy(df, c)
        assert res["custo_usado"] <= c["orcamento_maximo"], (
            f"Orçamento violado: {res['custo_usado']} > {c['orcamento_maximo']}"
        )

    def test_horas_nunca_excedidas(self, instancia_pequena):
        df, c = instancia_pequena
        res = solve_greedy(df, c)
        assert res["horas_usadas"] <= c["horas_maximas"], (
            f"Horas violadas: {res['horas_usadas']} > {c['horas_maximas']}"
        )

    def test_orcamento_nunca_excedido_grande(self, instancia_grande):
        df, c = instancia_grande
        res = solve_greedy(df, c)
        assert res["custo_usado"] <= c["orcamento_maximo"]

    def test_horas_nunca_excedidas_grande(self, instancia_grande):
        df, c = instancia_grande
        res = solve_greedy(df, c)
        assert res["horas_usadas"] <= c["horas_maximas"]

    def test_restricoes_muito_apertadas(self):
        """Quando o orçamento mal cobre um projeto, resultado deve ser factível."""
        df = generate_projects(20, seed=10)
        constraints = {"orcamento_maximo": 1, "horas_maximas": 1}
        res = solve_greedy(df, constraints)
        assert res["custo_usado"] <= 1
        assert res["horas_usadas"] <= 1

    def test_sem_projetos_disponiveis(self):
        """Restrições impossíveis → solução vazia e factível."""
        df = generate_projects(10, seed=5)
        constraints = {"orcamento_maximo": 0, "horas_maximas": 0}
        res = solve_greedy(df, constraints)
        assert res["n_selecionados"] == 0
        assert res["lucro_total"] == 0.0


class TestGreedyConsistency:
    """Os valores retornados devem ser internamente consistentes."""

    def test_lucro_total_consistente(self, instancia_pequena):
        df, c = instancia_pequena
        res = solve_greedy(df, c)
        lucro_calculado = df.loc[res["projetos_selecionados"], "lucro"].sum()
        assert abs(res["lucro_total"] - lucro_calculado) < 1e-6

    def test_custo_total_consistente(self, instancia_pequena):
        df, c = instancia_pequena
        res = solve_greedy(df, c)
        custo_calculado = df.loc[res["projetos_selecionados"], "custo"].sum()
        assert abs(res["custo_usado"] - custo_calculado) < 1e-6

    def test_n_selecionados_consistente(self, instancia_pequena):
        df, c = instancia_pequena
        res = solve_greedy(df, c)
        assert res["n_selecionados"] == len(res["projetos_selecionados"])

    def test_sem_projetos_duplicados(self, instancia_grande):
        df, c = instancia_grande
        res = solve_greedy(df, c)
        selecionados = res["projetos_selecionados"]
        assert len(selecionados) == len(set(selecionados)), "Projetos duplicados na solução"

    def test_indices_validos(self, instancia_grande):
        df, c = instancia_grande
        res = solve_greedy(df, c)
        for idx in res["projetos_selecionados"]:
            assert 0 <= idx < len(df), f"Índice inválido: {idx}"


class TestGreedyCamposRetorno:
    """O dicionário de retorno deve ter todos os campos esperados."""

    CAMPOS = [
        "metodo", "projetos_selecionados", "n_selecionados",
        "lucro_total", "custo_usado", "horas_usadas",
        "tempo_execucao", "status",
    ]

    def test_campos_presentes(self, instancia_pequena):
        df, c = instancia_pequena
        res = solve_greedy(df, c)
        for campo in self.CAMPOS:
            assert campo in res, f"Campo ausente: '{campo}'"

    def test_tempo_execucao_positivo(self, instancia_pequena):
        df, c = instancia_pequena
        res = solve_greedy(df, c)
        assert res["tempo_execucao"] >= 0

    def test_criterios_validos(self, instancia_pequena):
        df, c = instancia_pequena
        for criterion in ["eficiencia", "lucro", "custo"]:
            res = solve_greedy(df, c, criterion=criterion)
            assert res["custo_usado"] <= c["orcamento_maximo"]
            assert res["horas_usadas"] <= c["horas_maximas"]

    def test_criterio_invalido_levanta_erro(self, instancia_pequena):
        df, c = instancia_pequena
        with pytest.raises(ValueError):
            solve_greedy(df, c, criterion="invalido")
