# 📊 Otimização de Alocação de Recursos em Projetos

> Imagine que você tem vários projetos, mas não pode fazer todos. Essa aplicação ajuda a escolher quais valem mais a pena, considerando dinheiro e tempo disponíveis, comparando diferentes formas de tomar essa decisão.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit)](https://streamlit.io)
[![PuLP](https://img.shields.io/badge/PuLP-2.7%2B-green)](https://coin-or.github.io/pulp/)
[![Tests](https://img.shields.io/badge/tests-58%20passed-brightgreen)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🚀 Acesse o Dashboard Online

👉 **https://industrial-optimization-dashboard.streamlit.app/**

---

## Resumo Executivo

Projeto de otimização de alocação de recursos sob restrições de orçamento e horas de equipe, modelado como um problema de Mochila Multidimensional 0-1 (0-1 Multidimensional Knapsack Problem, NP-hard).

O projeto compara três abordagens — Programação Linear Inteira (método exato), Heurística Gulosa e Algoritmo Genético — quantificando o desempenho em termos de lucro obtido, tempo de execução e gap percentual em relação ao ótimo global, evidenciando o trade-off entre qualidade da solução e custo computacional.

Aplicável diretamente a cenários industriais, como seleção de projetos de P&D, priorização de investimentos e planejamento operacional.

> O trabalho demonstra, na prática, como diferentes estratégias de otimização se comportam em problemas combinatórios reais.

---

## Aplicações Industriais

| Setor | Problema análogo |
|-------|-----------------|
| P&D / Inovação | Selecionar projetos de pesquisa dentro do orçamento anual |
| Manufatura | Priorizar ordens de produção com capacidade de mão-de-obra limitada |
| Financeiro | Escolher investimentos com maior retorno dado um capital disponível |
| Logística | Alocar equipes e horas em rotas ou clientes de maior valor |
| TI / Produto | Priorizar features do backlog com time e sprint limitados |

---

## Dashboard Interativo

### Comparação dos métodos — cenário grande (100 projetos)
![Comparação](results/comparacao_grande.png)

### Convergência do Algoritmo Genético
![Convergência](results/convergencia_ag_grande.png)

### Projetos selecionados por método (heatmap — cenário pequeno)
![Heatmap](results/projetos_selecionados.png)

### Distribuição dos dados simulados
![Distribuição](results/distribuicao_projetos.png)

### Desempenho comparativo entre cenários
![Cenários](results/comparacao_cenarios.png)

---

## Formulação Matemática

**Variáveis de decisão:**

$$x_i \in \{0, 1\}, \quad i = 1, \ldots, n$$

**Maximizar:**

$$Z = \sum_{i=1}^{n} \text{lucro}_i \cdot x_i$$

**Sujeito a:**

$$\sum_{i=1}^{n} \text{custo}_i \cdot x_i \leq B \quad \text{(orçamento)}$$

$$\sum_{i=1}^{n} \text{horas}_i \cdot x_i \leq H \quad \text{(horas da equipe)}$$

$$x_i \in \{0, 1\} \quad \forall i$$

---

## Resultados Experimentais

Todos os experimentos foram realizados com `seed=42`, utilização de 55% dos recursos totais.

### Cenário Pequeno — 10 projetos · Orçamento R$ 268 · 187h

| Método | Lucro Total | Gap vs Ótimo | Tempo |
|--------|-------------|-------------|-------|
| PLI (Exato) | **R$ 844** | 0% | ~23 ms |
| Greedy | R$ 818 | 3,1% | < 1 ms |
| Genético (AG) | **R$ 844** | 0% | ~150 ms |

### Cenário Médio — 30 projetos · Orçamento R$ 975 · 457h

| Método | Lucro Total | Gap vs Ótimo | Tempo |
|--------|-------------|-------------|-------|
| PLI (Exato) | **R$ 2.551** | 0% | ~20 ms |
| Greedy | **R$ 2.551** | 0% | ~1 ms |
| Genético (AG) | **R$ 2.551** | 0% | ~620 ms |

### Cenário Grande — 100 projetos · Orçamento R$ 3.157 · 1.490h

| Método | Lucro Total | Gap vs Ótimo | Tempo |
|--------|-------------|-------------|-------|
| PLI (Exato) | **R$ 8.085** | 0% | ~125 ms |
| Greedy | R$ 7.998 | 1,1% | ~2 ms |
| Genético (AG) | R$ 8.077 | 0,1% | ~700 ms |

**Principais achados:**
- O PLI garante a solução ótima em todos os cenários com tempo inferior a 200 ms
- O Greedy é ~60× mais rápido que o PLI com perda média de 1–3%
- O AG encontra soluções quasi-ótimas (gap < 0,2%) no cenário grande

---

## Estrutura do Projeto

```
otimizacao-alocacao-recursos-python/
│
├── app.py                          # Dashboard Streamlit (interface web)
├── main.py                         # Script de linha de comando
├── requirements.txt
├── README.md
│
├── src/
│   ├── data/
│   │   └── data_generator.py       # Geração de dados simulados e cenários
│   ├── solvers/
│   │   ├── exact_solver.py         # PLI via PuLP + solver CBC
│   │   ├── greedy_solver.py        # Heurística gulosa por eficiência
│   │   └── genetic_solver.py       # Algoritmo Genético com reparo
│   └── visualization/
│       └── plots.py                # Gráficos matplotlib/seaborn
│
├── tests/                          # 58 testes unitários (pytest)
│   ├── test_data_generator.py
│   ├── test_exact_solver.py
│   ├── test_greedy_solver.py
│   └── test_genetic_solver.py
│
├── results/                        # 12 gráficos gerados automaticamente
└── .streamlit/
    └── config.toml                 # Tema do dashboard
```

---

## Como Rodar

### Pré-requisitos

- Python 3.9+

### Instalação

```bash
# 1. Clonar o repositório
git clone https://github.com/floresjacques26/otimizacao-alocacao-recursos-python.git
cd otimizacao-alocacao-recursos-python

# 2. Criar ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 3. Instalar dependências
pip install -r requirements.txt
```

### Dashboard Web

```bash
streamlit run app.py
```

Acesse **http://localhost:8501** no navegador.

### Terminal (sem interface)

```bash
python main.py
```

### Testes

```bash
pytest tests/ -v
```

---

## Análise de Complexidade

| Método | Complexidade | Otimalidade | Escalabilidade |
|--------|-------------|-------------|----------------|
| PLI (Branch & Bound) | NP-difícil | ✅ Garantida | Até ~500 variáveis |
| Greedy | O(n log n) | ❌ Nenhuma | ✅ Excelente |
| Algoritmo Genético | O(G × P × n) | ❌ Nenhuma | ✅ Boa (configurável) |

---

## Tecnologias

| Biblioteca | Uso |
|-----------|-----|
| [Streamlit](https://streamlit.io) | Interface web interativa |
| [PuLP](https://coin-or.github.io/pulp/) | Modelagem e solução PLI com solver CBC |
| [NumPy](https://numpy.org) | Operações vetorizadas no Algoritmo Genético |
| [Pandas](https://pandas.pydata.org) | Manipulação dos dados de projetos |
| [Matplotlib](https://matplotlib.org) | Visualizações e gráficos |
| [Seaborn](https://seaborn.pydata.org) | Heatmap de seleção de projetos |
| [pytest](https://pytest.org) | Testes unitários (58 testes) |

---

## Publicar Online (Streamlit Community Cloud)

1. Faça push para um repositório **público** no GitHub
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Conecte o repositório e selecione `app.py`
4. Clique em **Deploy** — link público em ~2 minutos

---

## Considerações Finais

### Hipótese
Algoritmos exatos encontram soluções ótimas para instâncias de tamanho moderado, enquanto heurísticas oferecem trade-offs competitivos de qualidade versus tempo em instâncias maiores.

### Metodologia
Dados gerados sinteticamente com distribuições uniformes; comparação controlada com seed fixo; três cenários com 10, 30 e 100 projetos; métrica principal: gap percentual em relação ao ótimo PLI.

### Limitações
- Dados sintéticos não capturam correlações reais entre variáveis de projeto
- O AG sem ajuste fino de hiperparâmetros pode não atingir sua capacidade máxima
- Instâncias acima de 500 projetos não foram avaliadas

### Trabalhos Futuros
- Implementação de GRASP e Simulated Annealing para comparação ampliada
- Otimização multi-objetivo (lucro × risco) com NSGA-II
- Dados reais de portfólios de projetos industriais
- Avaliação em instâncias de benchmark da literatura (OR-Library)

---

## 📚 Referências Teóricas e Técnicas

- Kellerer, H., Pferschy, U., & Pisinger, D. (2004). *Knapsack Problems*. Springer. — referência principal sobre o Problema da Mochila e suas variantes (0-1, multidimensional, fracionária), base teórica do problema modelado neste projeto.
- Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley. — obra fundacional de algoritmos genéticos; base para os operadores de seleção, cruzamento e mutação implementados.
- Mitchell, M. (1998). *An Introduction to Genetic Algorithms*. MIT Press. — introdução acessível à teoria evolutiva aplicada à otimização combinatória; utilizada como referência didática para o AG deste projeto.
- [Documentação PuLP](https://coin-or.github.io/pulp/) — biblioteca Python utilizada para modelagem e resolução do problema de Programação Linear Inteira via solver CBC (Branch & Bound).

---

## Possíveis Melhorias

- [ ] GRASP — Greedy Randomized Adaptive Search
- [ ] Simulated Annealing com cooling schedule adaptativo
- [ ] NSGA-II — otimização multi-objetivo (lucro × risco)
- [ ] Dependências entre projetos (restrições de precedência)
- [ ] Deploy automático com GitHub Actions
- [ ] Benchmark com instâncias da OR-Library

---

*Projeto desenvolvido como portfólio para pesquisa em otimização combinatória.*

---

## 👤 Autor

Alexandre Flores Jacques
🔗 LinkedIn: https://www.linkedin.com/in/alexandre-jacques-237857256
🔗 GitHub: https://github.com/floresjacques26
