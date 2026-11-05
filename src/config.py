"""Configurações centrais do pipeline: caminhos, datas de corte e definição dos eventos."""

from pathlib import Path

# Caminhos do projeto
RAIZ = Path(__file__).resolve().parents[1]
PASTA_DADOS = RAIZ / 'data'
PASTA_MODELOS = RAIZ / 'models'
PASTA_SAIDAS = RAIZ / 'outputs'

ARQUIVO_CONFIGS = PASTA_MODELOS / 'model_configs.json'       # Configurações escolhidas na etapa 4 (notebook)
ARQUIVO_MODELO = PASTA_MODELOS / 'modelo_final.pkl'          # Modelo campeão treinado pelo pipeline
ARQUIVO_METADADOS = PASTA_MODELOS / 'modelo_final.json'      # Informações sobre o modelo campeão

# Período útil da série (definido na EDA - etapa 2)
# - Antes de 05/01/2017 os registros são esparsos.
# - Depois de 19/08/2018 os pedidos estão incompletos na base disponibilizada pela Olist.
INICIO_SERIE = '2017-01-05'
FIM_SERIE = '2018-08-19'

# Previsão e validação
HORIZONTE = 28                 # Dias previstos a cada execução (4 semanas)
NUM_JANELAS_BACKTEST = 4       # Janelas de 28 dias usadas para comparar os modelos

# Eventos que alteram o padrão de vendas (etapa 3)
# - Eventos planejados são conhecidos com antecedência e entram também no período previsto.
# - Eventos não planejados só são conhecidos depois que acontecem: entram no treino e valem zero no futuro.
EVENTOS = ['feriado', 'black_friday', 'pos_black_friday', 'fim_de_ano', 'greve_caminhoneiros']
EVENTOS_NAO_PLANEJADOS = {
    'greve_caminhoneiros': [('2018-05-23', '2018-05-31')],
}
