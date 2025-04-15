import pandas as pd
from datetime import datetime
from ortools.sat.python import cp_model

# Ler as bases de dados
perdas_viagens = pd.read_excel('Case_exemplo.xlsx')
manutencao_veiculos = pd.read_excel('Dados_manutencao.xlsx')

# Ler a tabela de linhas que aceitam P-AR
regras_flexibilidade = pd.read_excel('base_Aceita_P-AR.xlsx')

# Renomear colunas para padronizar
perdas_viagens.rename(columns={
    'Linha ext': 'Linha_Ext',  # Coluna que contém a linha estendida
    'PENALIDADE': 'Penalidade'  # Coluna que contém o valor financeiro da perda
}, inplace=True)

manutencao_veiculos.rename(columns={
    'Veículo Proc': 'Veiculo',
    'Categoria': 'Categoria'  # Coluna que contém a categoria do carro
}, inplace=True)

# Converter formatos de data e hora
perdas_viagens['Data'] = pd.to_datetime(perdas_viagens['Data'], format='%d/%m/%Y')
perdas_viagens['H_Saida'] = pd.to_datetime(perdas_viagens['H_Saida'], format='%H:%M:%S').dt.time
manutencao_veiculos['Data'] = pd.to_datetime(manutencao_veiculos['Data'], format='%d/%m/%Y')

# Filtrar apenas as linhas que aceitam P-AR
linhas_aceitam_par = regras_flexibilidade[regras_flexibilidade['Aceita_P-AR'] == 'Sim']['Linha'].tolist()

# Função para alocar veículos às viagens usando OR-Tools
def alocar_carros(perdas_viagens, manutencao_veiculos, linhas_aceitam_par):
    # Adicionar coluna para o veículo alocado
    perdas_viagens['Carro_Alocado'] = None

    # Iterar sobre cada data única
    for data in perdas_viagens['Data'].unique():
        # Filtrar viagens e veículos disponíveis para o dia
        viagens_dia = perdas_viagens[perdas_viagens['Data'] == data].copy()
        carros_disponiveis = manutencao_veiculos[manutencao_veiculos['Data'] == data][['Veiculo', 'Categoria']].drop_duplicates()
        
        if carros_disponiveis.empty:
            print(f"Nenhum carro disponível em {data}. Pulando para o próximo dia.")
            continue  # Pular se não houver carros disponíveis
        
        print(f"\n--- Processando alocação para o dia {data} ---")
        print(f"Carros disponíveis: {carros_disponiveis}")
        print(f"Viagens no dia: {viagens_dia[['H_Saida', 'Linha_Ext', 'Penalidade', 'Categoria_Programada']]}")

        # Ordenar viagens por Penalidade (decrescente) e H_Saida (crescente)
        viagens_dia = viagens_dia.sort_values(by=['Penalidade', 'H_Saida'], ascending=[False, True])
        
        # Criar modelo de otimização
        model = cp_model.CpModel()

        # Variáveis de decisão
        num_viagens = len(viagens_dia)
        num_carros = len(carros_disponiveis)
        alocacao = {}
        for i in range(num_viagens):
            for j in range(num_carros):
                alocacao[(i, j)] = model.NewBoolVar(f'alocacao_viagem_{i}_carro_{j}')

        # Restrições
        # 1. Cada viagem pode ser alocada a no máximo um carro
        for i in range(num_viagens):
            model.Add(sum(alocacao[(i, j)] for j in range(num_carros)) <= 1)

        # 2. Cada carro só pode ser alocado a uma viagem por vez, com intervalo mínimo de 2 horas
        for j in range(num_carros):
            for i1 in range(num_viagens):
                for i2 in range(i1 + 1, num_viagens):
                    horario_viagem1 = datetime.combine(data, viagens_dia.iloc[i1]['H_Saida'])
                    horario_viagem2 = datetime.combine(data, viagens_dia.iloc[i2]['H_Saida'])
                    diferenca = abs((horario_viagem2 - horario_viagem1).total_seconds())
                    if diferenca < 7200:  # 2 horas em segundos
                        model.Add(alocacao[(i1, j)] + alocacao[(i2, j)] <= 1)

        # 3. Restrição de categoria: Um carro só pode ser alocado a uma viagem se a categoria for compatível
        for i in range(num_viagens):
            for j in range(num_carros):
                categoria_viagem = viagens_dia.iloc[i]['Categoria_Programada']  # Categoria da viagem
                categoria_carro = carros_disponiveis.iloc[j]['Categoria']  # Categoria do carro
                linha_viagem = viagens_dia.iloc[i]['Linha_Ext']  # Usar a coluna Linha_Ext para comparação

                # Verificar se a combinação é permitida
                if categoria_viagem != categoria_carro:
                    if categoria_carro == 'P-AR' and categoria_viagem == 'C-AR':
                        if linha_viagem in linhas_aceitam_par:
                            continue  # Permite a alocação
                    model.Add(alocacao[(i, j)] == 0)  # Impede a alocação

        # Função objetivo: Maximizar a soma das Penalidade das viagens alocadas
        objetivo = []
        for i in range(num_viagens):
            for j in range(num_carros):
                objetivo.append(viagens_dia.iloc[i]['Penalidade'] * alocacao[(i, j)])
        model.Maximize(sum(objetivo))

        # Resolver o modelo
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        # Verificar se a solução foi encontrada
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print(f"\nSolução encontrada para o dia {data}.")
            for i in range(num_viagens):
                for j in range(num_carros):
                    if solver.Value(alocacao[(i, j)]) == 1:
                        perdas_viagens.loc[viagens_dia.index[i], 'Carro_Alocado'] = carros_disponiveis.iloc[j]['Veiculo']
                        print(f"Viagem {i} (Hora: {viagens_dia.iloc[i]['H_Saida']}, Linha: {viagens_dia.iloc[i]['Linha_Ext']}, Penalidade: R${viagens_dia.iloc[i]['Penalidade']}) alocada ao carro {carros_disponiveis.iloc[j]['Veiculo']}.")
                        break
        else:
            print(f"Nenhuma solução viável encontrada para o dia {data}.")

    return perdas_viagens

# Executar função de alocação
perdas_viagens_alocadas = alocar_carros(perdas_viagens, manutencao_veiculos, linhas_aceitam_par)

# Exibir resultado
print("\n--- Resultado Final da Alocação ---")
print(perdas_viagens_alocadas[['Data', 'H_Saida', 'Linha_Ext', 'Carro', 'Penalidade', 'Carro_Alocado']])

# Salvar resultado em um arquivo Excel
perdas_viagens_alocadas.to_excel('perdas_viagens_alocadas.xlsx', index=False)