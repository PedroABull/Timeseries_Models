# Teste de Features 

import numpy as np
import pandas as pd
import pytest

from src import config, features, models


@pytest.fixture
def serie():
    #Série sintética com sazonalidade semanal, no mesmo formato da série real.
    datas = pd.date_range('2017-01-05', '2018-08-19', freq = 'D', name = 'sale_date')
    vendas = 200 + 50 * np.sin(2 * np.pi * datas.day_of_week / 7) + np.random.default_rng(42).normal(0, 10, len(datas))
    return features.preparar_serie(pd.DataFrame({'order_units' : vendas.round()}, index = datas))


def test_features_ml_nao_usam_vendas_do_periodo_previsto(serie):
    #As features dos 28 dias após a data de corte devem ser iguais com e sem as vendas desse período.
    corte = pd.Timestamp('2018-07-22')
    futuro = serie.loc[corte + pd.Timedelta(days = 1):].index[:config.HORIZONTE]

    vendas_sem_futuro = serie['order_units'].copy()
    vendas_sem_futuro.loc[futuro] = np.nan

    com_futuro = features.criar_features_ml(serie['order_units'], serie).loc[futuro]
    sem_futuro = features.criar_features_ml(vendas_sem_futuro, serie).loc[futuro]
    pd.testing.assert_frame_equal(com_futuro, sem_futuro)


def test_modelo_ml_prevê_sem_conhecer_o_futuro(serie):
    #A previsão do modelo de ML não pode mudar se as vendas após a data de corte forem alteradas.
    import xgboost as xgb
    corte = '2018-07-22'
    alterada = serie.copy()
    alterada.loc['2018-07-23':, 'order_units'] = 9999

    previsao_1 = models.ModeloML('XGBoost', xgb.XGBRegressor(n_estimators = 20, random_state = 42)).fit(serie.loc[:corte]).predict()
    previsao_2 = models.ModeloML('XGBoost', xgb.XGBRegressor(n_estimators = 20, random_state = 42)).fit(alterada.loc[:corte]).predict()
    pd.testing.assert_frame_equal(previsao_1, previsao_2)


def test_black_friday_calculada_pelo_calendario():
    assert features.black_friday(2017) == pd.Timestamp('2017-11-24')
    assert features.black_friday(2018) == pd.Timestamp('2018-11-23')


def test_eventos_nao_planejados_valem_zero_no_futuro():
    datas = pd.date_range('2018-05-20', '2018-06-05')
    assert features.criar_eventos(datas)['greve_caminhoneiros'].sum() == 9
    assert models.eventos_futuros(datas)['greve_caminhoneiros'].sum() == 0


def test_fim_de_ano_marca_somente_as_festas():
    eventos = features.criar_eventos(pd.date_range('2017-12-01', '2018-12-31'))
    marcados = eventos.index[eventos['fim_de_ano'] == 1]
    assert len(marcados) == 23     # 21/12/2017 a 01/01/2018 (12 dias) + 21/12/2018 a 31/12/2018 (11 dias)
    assert marcados.min() == pd.Timestamp('2017-12-21')
