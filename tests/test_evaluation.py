# Métricas de Backtest

import numpy as np
import pandas as pd
import pytest

from src import evaluation


def test_wmape_e_erro_total():
    real = np.array([100, 200, 300])
    previsto = np.array([110, 180, 300])
    metricas = evaluation.calcular_metricas(real, previsto)
    assert metricas['WMAPE (%)'] == pytest.approx(100 * 30 / 600)
    assert metricas['Erro total (%)'] == pytest.approx(100 * (590 / 600 - 1))
    assert metricas['MAE'] == pytest.approx(10)


def test_previsao_perfeita_tem_erro_zero():
    real = np.array([50, 60, 70])
    assert evaluation.calcular_metricas(real, real)['WMAPE (%)'] == 0


def test_janelas_de_backtest_terminam_na_data_de_corte():
    inicios = evaluation.inicio_janelas('2018-07-22', horizonte = 28, num_janelas = 4)
    assert list(inicios.strftime('%Y-%m-%d')) == ['2018-04-02', '2018-04-30', '2018-05-28', '2018-06-25']
    assert inicios[-1] + pd.Timedelta(days = 27) == pd.Timestamp('2018-07-22')
