# -*- coding: utf-8 -*-
"""
Testes unitários — Solver Heurístico: Algoritmo Genético.

Verificamos:
    1. Solução final sempre factível (restrições respeitadas)
    2. Histórico de convergência tem o tamanho correto
    3. O fitness nunca decresce ao longo das gerações (elitismo)
    4. Resultado é melhor ou igual ao Greedy na maioria dos casos
    5. Reprodutibilidade com mesma seed
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from data.data_generator import generate_projects, generate_constraints
from solvers.genetic_solver import solve_genetic


@pytest.fixture
def instancia_padrao():
    df = generate_projects(20, seed=42)
    constraints = generate_constraints(df, utilizacao=0.55)
    return df, constraints


class TestGeneticFeasibility:
    """A solução do AG deve sempre respeitar as restrições."""

    def test_orcamento_respeitado(self, instancia_padrao):
        df, c = instancia_padrao
        res = solve_genetic(df, c, pop_size=40, n_generations=50, seed=0)
        assert res["custo_usado"] <= c["orcamento_maximo"] + 1e-6, (
            f"Orçamento violado: {res['custo_usado']} > {c['orcamento_maximo']}"
        )

    def test_horas_respeitadas(self, instancia_padrao):
        df, c = instancia_padrao
        res = solve_genetic(df, c, pop_size=40, n_generations=50, seed=0)
        assert res["horas_usadas"] <= c["horas_maximas"] + 1e-6

    def test_restricoes_muito_apertadas(self):
        """Mesmo com restrições apertadas, a solução deve ser factível."""
        df = generate_projects(15, seed=5)
        constraints = {"orcamento_maximo": 10, "horas_maximas": 10}
        res = solve_genetic(df, constraints, pop_size=30, n_generations=30, seed=0)
        assert res["custo_usado"]  <= 10 + 1e-6
        assert res["horas_usadas"] <= 10 + 1e-6

    def test_sem_espaco_resultado_vazio(self):
        """Orçamento zero → solução vazia."""
        df = generate_projects(10, seed=1)
        constraints = {"orcamento_maximo": 0, "horas_maximas": 0}
        res = solve_genetic(df, constraints, pop_size=20, n_generations=20, seed=0)
        assert res["lucro_total"] == 0.0
        assert res["n_selecionados"] == 0


class TestGeneticConvergence:
    """Propriedades da curva de convergência."""

    def test_historico_tem_tamanho_correto(self, instancia_padrao):
        df, c = instancia_padrao
        n_gens = 80
        res = solve_genetic(df, c, pop_size=30, n_generations=n_gens, seed=0)
        assert len(res["historico_convergencia"]) == n_gens

    def test_fitness_nunca_decresce(self, instancia_padrao):
        """Elitismo garante que o melhor fitness nunca piora."""
        df, c = instancia_padrao
        res = solve_genetic(df, c, pop_size=50, n_generations=100, seed=42)
        hist = res["historico_convergencia"]
        for i in range(1, len(hist)):
            assert hist[i] >= hist[i-1] - 1e-6, (
                f"Fitness decresceu na geracao {i}: {hist[i-1]} -> {hist[i]}"
            )

    def test_historico_termina_no_melhor(self, instancia_padrao):
        """O último valor do histórico deve ser o lucro retornado."""
        df, c = instancia_padrao
        res = solve_genetic(df, c, pop_size=40, n_generations=60, seed=7)
        assert abs(res["historico_convergencia"][-1] - res["lucro_total"]) < 1e-6


class TestGeneticConsistency:
    """Consistência interna dos valores retornados."""

    def test_lucro_consistente_com_projetos(self, instancia_padrao):
        df, c = instancia_padrao
        res = solve_genetic(df, c, pop_size=40, n_generations=50, seed=0)
        lucro_calc = sum(df["lucro"].iloc[i] for i in res["projetos_selecionados"])
        assert abs(res["lucro_total"] - lucro_calc) < 1e-6

    def test_n_selecionados_consistente(self, instancia_padrao):
        df, c = instancia_padrao
        res = solve_genetic(df, c, pop_size=40, n_generations=50, seed=0)
        assert res["n_selecionados"] == len(res["projetos_selecionados"])

    def test_reprodutibilidade(self, instancia_padrao):
        """Mesma seed deve produzir o mesmo resultado."""
        df, c = instancia_padrao
        res1 = solve_genetic(df, c, pop_size=40, n_generations=50, seed=123)
        res2 = solve_genetic(df, c, pop_size=40, n_generations=50, seed=123)
        assert res1["lucro_total"] == res2["lucro_total"]
        assert res1["projetos_selecionados"] == res2["projetos_selecionados"]

    def test_seeds_diferentes_podem_divergir(self, instancia_padrao):
        """Seeds diferentes podem (não obrigatoriamente) dar resultados diferentes."""
        df, c = instancia_padrao
        res1 = solve_genetic(df, c, pop_size=20, n_generations=30, seed=1)
        res2 = solve_genetic(df, c, pop_size=20, n_generations=30, seed=9999)
        # Apenas verifica que ambos são factíveis — não exige que difiram
        assert res1["custo_usado"] <= c["orcamento_maximo"] + 1e-6
        assert res2["custo_usado"] <= c["orcamento_maximo"] + 1e-6

    def test_campos_obrigatorios(self, instancia_padrao):
        df, c = instancia_padrao
        res = solve_genetic(df, c, pop_size=30, n_generations=30, seed=0)
        campos = ["metodo", "projetos_selecionados", "n_selecionados",
                  "lucro_total", "custo_usado", "horas_usadas",
                  "tempo_execucao", "status", "historico_convergencia"]
        for campo in campos:
            assert campo in res, f"Campo ausente: '{campo}'"


class TestGeneticQuality:
    """O AG deve encontrar soluções de qualidade razoável."""

    def test_lucro_positivo(self, instancia_padrao):
        df, c = instancia_padrao
        res = solve_genetic(df, c, pop_size=60, n_generations=100, seed=42)
        assert res["lucro_total"] > 0

    def test_gap_maximo_aceitavel_vs_exato(self):
        """AG deve chegar a no máximo 5% abaixo do ótimo PLI com config generosa."""
        from solvers.exact_solver import solve_exact
        df = generate_projects(20, seed=42)
        c  = generate_constraints(df, utilizacao=0.55)
        res_exato = solve_exact(df, c)
        res_ag    = solve_genetic(df, c, pop_size=100, n_generations=300, seed=42)
        if res_exato["lucro_total"] > 0:
            gap = (res_exato["lucro_total"] - res_ag["lucro_total"]) / res_exato["lucro_total"]
            assert gap <= 0.05, f"AG ficou {gap*100:.1f}% abaixo do otimo (limite: 5%)"
