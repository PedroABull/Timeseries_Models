"""Métricas e validação com origem móvel (backtest) - etapa 4."""

import numpy as np
import pandas as pd

from src import config


def calcular_metricas(real, previsto):
    real, previsto = np.asarray(real, dtype = float), np.asarray(previsto, dtype = float)
    erro = real - previsto
    return {'MAE' : np.abs(erro).mean(),
            'RMSE' : np.sqrt((erro ** 2).mean()),
            'WMAPE (%)' : 100 * np.abs(erro).sum() / real.sum(),
            'Erro total (%)' : 100 * (previsto.sum() / real.sum() - 1)}


def inicio_janelas(data_corte, horizonte=config.HORIZONTE, num_janelas=config.NUM_JANELAS_BACKTEST):
    """Início das janelas de backtest: janelas consecutivas de `horizonte` dias, a última terminando na data de corte."""
    inicio_ultima = pd.Timestamp(data_corte) - pd.Timedelta(days = horizonte - 1)
    return pd.date_range(end = inicio_ultima, periods = num_janelas, freq = f'{horizonte}D')


def backtest(criar_modelo, df, inicios, horizonte=config.HORIZONTE):
    """Treina o modelo com o histórico anterior a cada janela, prevê a janela e calcula as métricas."""
    resultados, previsoes = [], []
    for inicio in inicios:
        modelo = criar_modelo().fit(df.loc[:inicio - pd.Timedelta(days = 1)])
        previsto = modelo.predict(horizonte)['previsao']
        real = df.loc[previsto.index, 'order_units']
        resultados.append({'inicio_janela' : inicio.date(), **calcular_metricas(real, previsto)})
        previsoes.append(previsto)
    return pd.DataFrame(resultados), pd.concat(previsoes)
