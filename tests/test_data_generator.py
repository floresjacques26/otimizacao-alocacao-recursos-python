"""
Testes unitários — módulo de geração de dados.
Execute com:  pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import pandas as pd
from data.data_generator import generate_projects, generate_constraints, get_scenarios


class TestGenerateProjects:
    def test_retorna_dataframe(self):
        df = generate_projects(10)
        assert isinstance(df, pd.DataFrame)

    def test_numero_correto_de_projetos(self):
        for n in [5, 10, 30, 100]:
            df = generate_projects(n)
            assert len(df) == n

    def test_colunas_presentes(self):
        df = generate_projects(10)
        for col in ["projeto_id", "nome", "custo", "horas", "lucro", "eficiencia"]:
            assert col in df.columns, f"Coluna '{col}' ausente"

    def test_valores_positivos(self):
        df = generate_projects(50, seed=0)
        assert (df["custo"]  > 0).all(), "Custos devem ser positivos"
        assert (df["horas"]  > 0).all(), "Horas devem ser positivas"
        assert (df["lucro"]  > 0).all(), "Lucros devem ser positivos"

    def test_eficiencia_calculada_corretamente(self):
        df = generate_projects(20, seed=42)
        eficiencia_esperada = df["lucro"] / (df["custo"] + df["horas"])
        assert (abs(df["eficiencia"] - eficiencia_esperada) < 1e-9).all()

    def test_reproducibilidade_com_seed(self):
        df1 = generate_projects(15, seed=7)
        df2 = generate_projects(15, seed=7)
        assert df1["lucro"].tolist() == df2["lucro"].tolist()

    def test_seeds_diferentes_geram_dados_diferentes(self):
        df1 = generate_projects(15, seed=1)
        df2 = generate_projects(15, seed=2)
        assert df1["lucro"].tolist() != df2["lucro"].tolist()

    def test_intervalo_custo_respeitado(self):
        df = generate_projects(100, custo_range=(20, 50), seed=42)
        assert df["custo"].min() >= 20
        assert df["custo"].max() <= 50

    def test_intervalo_lucro_respeitado(self):
        df = generate_projects(100, lucro_range=(50, 150), seed=42)
        assert df["lucro"].min() >= 50
        assert df["lucro"].max() <= 150


class TestGenerateConstraints:
    def setup_method(self):
        self.df = generate_projects(20, seed=42)

    def test_retorna_chaves_esperadas(self):
        c = generate_constraints(self.df)
        assert "orcamento_maximo" in c
        assert "horas_maximas" in c

    def test_restricoes_positivas(self):
        c = generate_constraints(self.df)
        assert c["orcamento_maximo"] > 0
        assert c["horas_maximas"] > 0

    def test_utilizacao_padrao_menor_que_total(self):
        c = generate_constraints(self.df, utilizacao=0.55)
        assert c["orcamento_maximo"] < self.df["custo"].sum()
        assert c["horas_maximas"]    < self.df["horas"].sum()

    def test_utilizacao_100_porcento(self):
        c = generate_constraints(self.df, utilizacao=1.0)
        assert c["orcamento_maximo"] == int(self.df["custo"].sum())

    def test_utilizacao_proporcional(self):
        c = generate_constraints(self.df, utilizacao=0.40)
        esperado = int(self.df["custo"].sum() * 0.40)
        assert c["orcamento_maximo"] == esperado


class TestGetScenarios:
    def test_retorna_tres_cenarios(self):
        s = get_scenarios()
        assert len(s) == 3

    def test_cenarios_esperados(self):
        s = get_scenarios()
        for nome in ["pequeno", "medio", "grande"]:
            assert nome in s

    def test_n_projetos_crescente(self):
        s = get_scenarios()
        assert s["pequeno"]["n_projetos"] < s["medio"]["n_projetos"] < s["grande"]["n_projetos"]
