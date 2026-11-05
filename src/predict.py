"""Gera a previsão de vendas dos próximos dias com o modelo salvo pelo train.py.

Uso:
    python -m src.predict
    python -m src.predict --horizonte 14
"""

import argparse
import json
import logging

import joblib

from src import config

logger = logging.getLogger(__name__)


def prever(horizonte=config.HORIZONTE):
    modelo = joblib.load(config.ARQUIVO_MODELO)
    with open(config.ARQUIVO_METADADOS, encoding = 'utf-8') as arquivo:
        metadados = json.load(arquivo)
    logger.info('Modelo %s treinado com dados até %s', metadados['modelo'], metadados['data_corte'])

    previsao = modelo.predict(horizonte).round(1)
    config.PASTA_SAIDAS.mkdir(exist_ok = True)
    inicio, fim = previsao.index.min().date(), previsao.index.max().date()
    caminho = config.PASTA_SAIDAS / f'previsao_{inicio}_{fim}.csv'
    previsao.to_csv(caminho)
    logger.info('Previsão de %s a %s salva em %s', inicio, fim, caminho.name)
    return previsao


def main():
    parser = argparse.ArgumentParser(description = 'Gera a previsão de vendas diárias com o modelo treinado.')
    parser.add_argument('--horizonte', type = int, default = config.HORIZONTE, help = 'Número de dias a prever.')
    args = parser.parse_args()
    logging.basicConfig(level = logging.INFO, format = '%(asctime)s | %(levelname)s | %(message)s')
    previsao = prever(args.horizonte)
    print('\nPrevisão semanal (unidades):')
    print(previsao['previsao'].resample('W-SUN').sum().round(0).to_string())     # Os limites do intervalo não são somados: a soma dos limites diários não é o limite semanal


if __name__ == '__main__':
    main()
