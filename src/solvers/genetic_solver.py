"""
Solver Heurístico: Algoritmo Genético (AG).

─────────────────────────────────────────────────────────────────────────────
INSPIRAÇÃO BIOLÓGICA
─────────────────────────────────────────────────────────────────────────────

O Algoritmo Genético imita a evolução natural:
    - Uma "população" de soluções candidatas evolui ao longo de "gerações"
    - Soluções melhores têm mais chance de se reproduzir (seleção natural)
    - Filhos herdam características dos pais (cruzamento)
    - Pequenas mutações introduzem diversidade genética

─────────────────────────────────────────────────────────────────────────────
REPRESENTAÇÃO E COMPONENTES
─────────────────────────────────────────────────────────────────────────────

Codificação (cromossomo):
    Vetor binário de tamanho n (número de projetos)
    cromossomo[i] = 1  →  projeto i SELECIONADO
    cromossomo[i] = 0  →  projeto i NÃO selecionado

    Exemplo com 5 projetos: [1, 0, 1, 1, 0] → projetos 0, 2 e 3 selecionados

Função de fitness:
    fitness = lucro_total   se a solução for factível (respeita restrições)
    fitness = 0             se violar qualquer restrição (após reparo)

Operadores genéticos:
    ┌─ Seleção:    Torneio — k indivíduos disputam, o melhor vence
    ├─ Cruzamento: Ponto único — divide dois pais e troca os segmentos
    ├─ Mutação:    Bit-flip — cada gene tem probabilidade p de ser invertido
    └─ Elitismo:   O melhor indivíduo sempre passa para a próxima geração

Reparo de infactibilidade:
    Se após cruzamento/mutação a solução violar as restrições,
    removemos projetos de menor eficiência até ela se tornar factível.
    Isso é mais eficaz do que apenas penalizar o fitness.

Complexidade:
    O(G × P × n)  onde G = gerações, P = tamanho da população, n = projetos
─────────────────────────────────────────────────────────────────────────────
"""

import time

import numpy as np
import pandas as pd


def _reparar(cromossomo: np.ndarray, custos, horas, lucros, orcamento_max, horas_max) -> np.ndarray:
    """
    Torna uma solução infactível factível removendo projetos de menor eficiência.

    Estratégia: ordenar os projetos selecionados pela razão lucro/(custo+horas)
    e remover o menos eficiente primeiro até as restrições serem satisfeitas.
    """
    crom = cromossomo.copy()

    # Enquanto infactível, remover o projeto selecionado de menor eficiência
    while True:
        custo_total = np.dot(crom, custos)
        horas_total = np.dot(crom, horas)

        if custo_total <= orcamento_max and horas_total <= horas_max:
            break  # solução já é factível

        idx_selecionados = np.where(crom == 1)[0]
        if len(idx_selecionados) == 0:
            break  # solução vazia, nada mais a remover

        # Calcular eficiência dos projetos selecionados
        eficiencias = lucros[idx_selecionados] / (custos[idx_selecionados] + horas[idx_selecionados] + 1e-9)

        # Remover o de menor eficiência
        pior = idx_selecionados[np.argmin(eficiencias)]
        crom[pior] = 0

    return crom


def solve_genetic(
    df: pd.DataFrame,
    constraints: dict,
    pop_size: int = 100,
    n_generations: int = 200,
    crossover_rate: float = 0.85,
    mutation_rate: float = 0.02,
    tournament_size: int = 5,
    seed: int = 42,
) -> dict:
    """
    Resolve o problema de alocação usando Algoritmo Genético.

    Parâmetros:
        df             : DataFrame de projetos
        constraints    : dict com 'orcamento_maximo' e 'horas_maximas'
        pop_size       : tamanho da população (indivíduos por geração)
        n_generations  : número máximo de gerações
        crossover_rate : probabilidade de cruzamento entre dois pais
        mutation_rate  : probabilidade de mutação por gene (bit-flip)
        tournament_size: número de competidores na seleção por torneio
        seed           : semente aleatória para reprodutibilidade

    Retorna:
        dict com campos compatíveis com os demais solvers + histórico de convergência.
    """
    rng = np.random.default_rng(seed)

    n = len(df)

    # Extrair arrays numpy para operações vetorizadas eficientes
    custos = df["custo"].values.astype(float)
    horas = df["horas"].values.astype(float)
    lucros = df["lucro"].values.astype(float)

    orcamento_max = constraints["orcamento_maximo"]
    horas_max = constraints["horas_maximas"]

    inicio = time.perf_counter()

    # ── Funções internas do AG ────────────────────────────────────────────

    def calcular_fitness(crom: np.ndarray) -> float:
        """
        Calcula o fitness de um cromossomo.
        Fitness = lucro total se factível, caso contrário 0 (após reparo).
        """
        crom_reparado = _reparar(crom, custos, horas, lucros, orcamento_max, horas_max)
        return float(np.dot(crom_reparado, lucros))

    def selecao_torneio(populacao: np.ndarray, fitness_vals: np.ndarray) -> np.ndarray:
        """
        Seleciona um indivíduo via torneio:
        sorteia k candidatos aleatórios e retorna o de maior fitness.
        """
        candidatos = rng.choice(len(populacao), size=tournament_size, replace=False)
        melhor_idx = candidatos[np.argmax(fitness_vals[candidatos])]
        return populacao[melhor_idx].copy()

    def cruzamento_ponto_unico(pai1: np.ndarray, pai2: np.ndarray):
        """
        Cruzamento de ponto único:
        um ponto aleatório divide os pais, e os segmentos são trocados.

        Antes:  pai1 = [A A A | B B B]
                pai2 = [C C C | D D D]
        Depois: filho1 = [A A A | D D D]
                filho2 = [C C C | B B B]
        """
        if rng.random() < crossover_rate and n > 1:
            ponto = rng.integers(1, n)  # ponto de corte (não nos extremos)
            filho1 = np.concatenate([pai1[:ponto], pai2[ponto:]])
            filho2 = np.concatenate([pai2[:ponto], pai1[ponto:]])
            return filho1, filho2
        # Sem cruzamento: retornar cópias dos pais
        return pai1.copy(), pai2.copy()

    def mutacao_bit_flip(crom: np.ndarray) -> np.ndarray:
        """
        Mutação bit-flip: cada gene é invertido (0→1 ou 1→0)
        com probabilidade mutation_rate.
        """
        mascara = rng.random(n) < mutation_rate
        return np.where(mascara, 1 - crom, crom).astype(int)

    # ── Inicialização da população ────────────────────────────────────────
    # Cada indivíduo é um vetor binário aleatório de tamanho n
    populacao = rng.integers(0, 2, size=(pop_size, n))

    # Reparar toda a população inicial (garantir factibilidade desde o início)
    populacao = np.array([
        _reparar(ind, custos, horas, lucros, orcamento_max, horas_max)
        for ind in populacao
    ])

    melhor_global = None
    melhor_fitness_global = -1.0
    historico_convergencia = []  # melhor fitness por geração (para o gráfico)

    # ── Loop evolutivo principal ──────────────────────────────────────────
    for geracao in range(n_generations):

        # Calcular fitness de toda a população
        fitness_vals = np.array([calcular_fitness(ind) for ind in populacao])

        # Identificar e guardar o melhor desta geração
        idx_melhor = int(np.argmax(fitness_vals))
        if fitness_vals[idx_melhor] > melhor_fitness_global:
            melhor_fitness_global = fitness_vals[idx_melhor]
            melhor_global = populacao[idx_melhor].copy()

        historico_convergencia.append(melhor_fitness_global)

        # ── Construir nova população ──────────────────────────────────────

        # Elitismo: o melhor indivíduo sobrevive intacto para a próxima geração
        nova_populacao = [melhor_global.copy()]

        while len(nova_populacao) < pop_size:
            # Seleção: escolher dois pais por torneio
            pai1 = selecao_torneio(populacao, fitness_vals)
            pai2 = selecao_torneio(populacao, fitness_vals)

            # Cruzamento: gerar dois filhos
            filho1, filho2 = cruzamento_ponto_unico(pai1, pai2)

            # Mutação: aplicar perturbação aleatória
            filho1 = mutacao_bit_flip(filho1)
            filho2 = mutacao_bit_flip(filho2)

            # Reparo: garantir que os filhos sejam factíveis
            filho1 = _reparar(filho1, custos, horas, lucros, orcamento_max, horas_max)
            filho2 = _reparar(filho2, custos, horas, lucros, orcamento_max, horas_max)

            nova_populacao.extend([filho1, filho2])

        # Truncar para manter tamanho fixo da população
        populacao = np.array(nova_populacao[:pop_size])

    tempo = time.perf_counter() - inicio

    # ── Extrair resultado final ───────────────────────────────────────────
    selecionados = [int(i) for i in range(n) if melhor_global[i] == 1]
    lucro_total = float(np.dot(melhor_global, lucros))
    custo_usado = float(np.dot(melhor_global, custos))
    horas_usadas = float(np.dot(melhor_global, horas))

    return {
        "metodo": "Genético (AG)",
        "projetos_selecionados": selecionados,
        "n_selecionados": len(selecionados),
        "lucro_total": lucro_total,
        "custo_usado": custo_usado,
        "horas_usadas": horas_usadas,
        "tempo_execucao": tempo,
        "status": "Heurístico",
        "gap": None,
        "historico_convergencia": historico_convergencia,
        "parametros": {
            "pop_size": pop_size,
            "n_generations": n_generations,
            "crossover_rate": crossover_rate,
            "mutation_rate": mutation_rate,
            "tournament_size": tournament_size,
        },
    }
