"""Simulação do uso do modelo em produção, com retreino e monitoramento semanais.

A cada semana (domingo), o modelo é treinado com os dados disponíveis até aquele dia e prevê as 4 semanas
seguintes. Na semana seguinte, as vendas reais são comparadas com a previsão: se o WMAPE da semana passar do
limite definido, um alerta é registrado para investigação (por exemplo, um evento não previsto).

Uso:
    python -m src.simulate_production --inicio 2018-04-01 --limite-alerta 25
"""

import argparse
import json
import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from src import config, data, evaluation, features, models

logger = logging.getLogger(__name__)


def simular(inicio, fim=config.FIM_SERIE, limite_alerta=25.0, nome_modelo=None):
    with open(config.ARQUIVO_CONFIGS) as arquivo:
        configs = json.load(arquivo)
    if nome_modelo is None:
        with open(config.ARQUIVO_METADADOS, encoding = 'utf-8') as arquivo:
            nome_modelo = json.load(arquivo)['modelo']
    criar_modelo = models.criar_modelos(configs)[nome_modelo]
    df = features.preparar_serie(data.construir_serie_diaria())

    # Execuções semanais: cada origem é o último dia de dados disponível no momento da previsão
    origens = pd.date_range(inicio, pd.Timestamp(fim) - pd.Timedelta(days = 7), freq = '7D')
    logger.info('Simulando %d semanas com o modelo %s', len(origens), nome_modelo)

    monitoramento, previsoes_semana = [], []
    for origem in origens:
        previsao = criar_modelo().fit(df.loc[:origem]).predict(config.HORIZONTE)['previsao']

        # Semana seguinte à execução: é a semana em que a previsão é usada e depois comparada com o real
        semana = previsao.iloc[:7]
        real = df.loc[semana.index, 'order_units']
        metricas = evaluation.calcular_metricas(real, semana)
        alerta = metricas['WMAPE (%)'] > limite_alerta
        monitoramento.append({'execucao' : origem.date(),
                              'semana_prevista' : f'{semana.index.min().date()} a {semana.index.max().date()}',
                              'vendas_reais' : int(real.sum()),
                              'vendas_previstas' : round(semana.sum()),
                              'WMAPE (%)' : round(metricas['WMAPE (%)'], 2),
                              'Erro total (%)' : round(metricas['Erro total (%)'], 2),
                              'alerta' : alerta})
        previsoes_semana.append(semana)
        if alerta:
            logger.warning('Semana %s: WMAPE de %.1f%% acima do limite de %.0f%% - investigar', semana.index.min().date(), metricas['WMAPE (%)'], limite_alerta)

    monitoramento = pd.DataFrame(monitoramento)
    previsoes_semana = pd.concat(previsoes_semana)
    salvar_resultados(df, monitoramento, previsoes_semana, limite_alerta, nome_modelo)
    return monitoramento


def salvar_resultados(df, monitoramento, previsoes_semana, limite_alerta, nome_modelo):
    config.PASTA_SAIDAS.mkdir(exist_ok = True)
    monitoramento.to_csv(config.PASTA_SAIDAS / 'monitoramento.csv', index = False)

    fig, axs = plt.subplots(nrows = 2, ncols = 1, figsize = (20, 9), tight_layout = True)
    axs[0].plot(df.loc[previsoes_semana.index, 'order_units'], label = 'Real', color = 'orange', linewidth = 2)
    axs[0].plot(previsoes_semana, label = f'Previsão ({nome_modelo}, retreino semanal)', color = 'green')
    axs[0].set_title('Vendas diárias em unidades - simulação de produção')
    axs[0].legend()

    cores = ['red' if alerta else 'navy' for alerta in monitoramento['alerta']]
    axs[1].bar(monitoramento['semana_prevista'].str[:10], monitoramento['WMAPE (%)'], color = cores)
    axs[1].axhline(limite_alerta, color = 'red', linestyle = '--', label = f'Limite de alerta ({limite_alerta:.0f}%)')
    axs[1].set_title('Monitoramento - WMAPE de cada semana (em vermelho, semanas com alerta)')
    axs[1].tick_params(axis = 'x', rotation = 45)
    axs[1].legend()
    fig.savefig(config.PASTA_SAIDAS / 'monitoramento.png', dpi = 100)
    plt.close(fig)
    logger.info('Resultados salvos em outputs/monitoramento.csv e outputs/monitoramento.png')


def main():
    parser = argparse.ArgumentParser(description = 'Simula o uso do modelo em produção com retreino e monitoramento semanais.')
    parser.add_argument('--inicio', default = '2018-04-01', help = 'Data da primeira execução (AAAA-MM-DD).')
    parser.add_argument('--fim', default = config.FIM_SERIE, help = 'Última data com vendas reais disponíveis.')
    parser.add_argument('--limite-alerta', type = float, default = 25.0, help = 'WMAPE semanal (%%) acima do qual um alerta é gerado.')
    parser.add_argument('--modelo', default = None, help = 'Nome do modelo (padrão: modelo campeão salvo pelo train.py).')
    args = parser.parse_args()
    logging.basicConfig(level = logging.INFO, format = '%(asctime)s | %(levelname)s | %(message)s')
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    monitoramento = simular(args.inicio, args.fim, args.limite_alerta, args.modelo)
    print('\nMonitoramento semanal:')
    print(monitoramento.to_string(index = False))
    print('\nWMAPE médio das semanas: {:.2f}% | semanas com alerta: {} de {}'.format(
        monitoramento['WMAPE (%)'].mean(), monitoramento['alerta'].sum(), len(monitoramento)))


if __name__ == '__main__':
    main()
