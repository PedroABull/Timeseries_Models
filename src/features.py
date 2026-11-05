"""Variáveis de eventos e features dos modelos (etapa 3)."""

import holidays
import numpy as np
import pandas as pd

from src import config


def black_friday(ano):
    """Black Friday: sexta-feira seguinte à quarta quinta-feira de novembro."""
    quintas = pd.date_range(f'{ano}-11-01', f'{ano}-11-30', freq = 'W-THU')
    return quintas[3] + pd.Timedelta(days = 1)


def criar_eventos(datas, incluir_nao_planejados=True):
    """Cria as variáveis de eventos para um conjunto de datas.

    Os eventos planejados são calculados por regras de calendário, o que permite gerá-los para qualquer
    ano futuro. Os eventos não planejados vêm da lista do config e, no período previsto, devem valer zero
    (incluir_nao_planejados=False), já que não eram conhecidos no momento da previsão.
    """
    datas = pd.DatetimeIndex(datas)
    eventos = pd.DataFrame(0, index = datas, columns = config.EVENTOS)

    feriados = holidays.Brazil(years = sorted(datas.year.unique()))
    eventos['feriado'] = datas.isin(pd.to_datetime(list(feriados.keys()))).astype(int)

    for ano in range(datas.year.min() - 1, datas.year.max() + 1):
        bf = black_friday(ano)
        eventos.loc[datas == bf, 'black_friday'] = 1
        eventos.loc[(datas > bf) & (datas <= bf + pd.Timedelta(days = 4)), 'pos_black_friday'] = 1
        eventos.loc[(datas >= f'{ano}-12-21') & (datas <= f'{ano + 1}-01-01'), 'fim_de_ano'] = 1     # Festas de fim de ano

    if incluir_nao_planejados:
        for evento, periodos in config.EVENTOS_NAO_PLANEJADOS.items():
            for inicio, fim in periodos:
                eventos.loc[(datas >= inicio) & (datas <= fim), evento] = 1

    return eventos


def preparar_serie(serie):
    """Acrescenta à série de vendas o logaritmo das vendas e as variáveis de eventos."""
    df = serie[['order_units']].copy()
    df['log_units'] = np.log(df['order_units'])
    return df.join(criar_eventos(df.index))


def criar_features_ml(vendas, eventos, horizonte=config.HORIZONTE):
    """Cria as features de machine learning: calendário, eventos e vendas passadas com defasagem >= horizonte.

    `vendas` pode conter NaN nas datas futuras: como todas as defasagens são de pelo menos `horizonte` dias,
    as features dessas datas dependem somente de vendas conhecidas.
    """
    features = pd.DataFrame(index = vendas.index)
    features['month'] = vendas.index.month
    features['day_of_week'] = vendas.index.day_of_week
    features['day_of_month'] = vendas.index.day
    features['lag_28'] = vendas.shift(horizonte)
    features['lag_35'] = vendas.shift(horizonte + 7)
    features['lag_42'] = vendas.shift(horizonte + 14)
    features['rolling_mean_7_lag28'] = vendas.shift(horizonte).rolling(7).mean()
    features['rolling_mean_28_lag28'] = vendas.shift(horizonte).rolling(28).mean()
    return features.join(eventos[config.EVENTOS])
