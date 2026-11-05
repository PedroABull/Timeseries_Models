"""Treino do modelo de previsão.

Executa o backtest de todos os modelos candidatos com os dados até a data de corte, escolhe o modelo com o
menor WMAPE médio, treina o campeão com todo o histórico e salva o modelo em models.

Uso:
    python -m src.train --cutoff 2018-07-22
"""

import argparse
import json
import logging
from datetime import datetime

import joblib
import pandas as pd

from src import config, data, evaluation, features, models

logger = logging.getLogger(__name__)


def treinar(data_corte=config.FIM_SERIE):
    data_corte = pd.Timestamp(data_corte)
    with open(config.ARQUIVO_CONFIGS) as arquivo:
        configs = json.load(arquivo)

    # Dados disponíveis até a data de corte
    df = features.preparar_serie(data.construir_serie_diaria()).loc[:data_corte]
    logger.info('Treinando com dados até %s (%d dias)', data_corte.date(), len(df))

    # Backtest de todos os candidatos
    inicios = evaluation.inicio_janelas(data_corte)
    logger.info('Janelas de backtest: %s', ', '.join(str(i.date()) for i in inicios))
    candidatos = models.criar_modelos(configs)
    relatorio = []
    for nome, criar_modelo in candidatos.items():
        resultado, _ = evaluation.backtest(criar_modelo, df, inicios)
        media = resultado.drop(columns = 'inicio_janela').mean()
        relatorio.append({'modelo' : nome, **media.round(2).to_dict()})
        logger.info('%-15s WMAPE médio = %.2f%%', nome, media['WMAPE (%)'])
    relatorio = pd.DataFrame(relatorio).sort_values('WMAPE (%)').reset_index(drop = True)

    # Modelo campeão treinado com todo o histórico
    campeao = relatorio.loc[0, 'modelo']
    modelo = candidatos[campeao]().fit(df)
    logger.info('Modelo campeão: %s', campeao)

    # Salvando o modelo, os metadados e o relatório do backtest
    config.PASTA_SAIDAS.mkdir(exist_ok = True)
    joblib.dump(modelo, config.ARQUIVO_MODELO)
    metadados = {'modelo' : campeao,
                 'data_corte' : str(data_corte.date()),
                 'treinado_em' : datetime.now().isoformat(timespec = 'seconds'),
                 'janelas_backtest' : [str(i.date()) for i in inicios],
                 'metricas_backtest' : relatorio.loc[0].drop('modelo').to_dict(),
                 'configuracoes' : configs}
    with open(config.ARQUIVO_METADADOS, 'w', encoding = 'utf-8') as arquivo:
        json.dump(metadados, arquivo, indent = 4, ensure_ascii = False)
    caminho_relatorio = config.PASTA_SAIDAS / f'backtest_{data_corte.date()}.csv'
    relatorio.to_csv(caminho_relatorio, index = False)
    logger.info('Modelo salvo em %s | relatório em %s', config.ARQUIVO_MODELO.name, caminho_relatorio.name)
    return modelo, relatorio


def main():
    parser = argparse.ArgumentParser(description = 'Treina o modelo de previsão de vendas diárias.')
    parser.add_argument('--cutoff', default = config.FIM_SERIE, help = 'Última data de dados usada no treino (AAAA-MM-DD).')
    args = parser.parse_args()
    logging.basicConfig(level = logging.INFO, format = '%(asctime)s | %(levelname)s | %(message)s')
    _, relatorio = treinar(args.cutoff)
    print('\nResultado do backtest (média das janelas):')
    print(relatorio.to_string(index = False))


if __name__ == '__main__':
    main()
