"""Modelos de previsão com uma interface comum (etapas 4 e 5).

Todos os modelos seguem o mesmo padrão:
- fit(df): treina com a série preparada por features.preparar_serie (vendas, log e eventos) até a última data de df.
- predict(horizonte): prevê os `horizonte` dias seguintes à última data de treino e retorna um DataFrame
  com a coluna 'previsao' (e, quando disponível, os limites do intervalo de previsão de 80%).
"""

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src import config, features


def datas_futuras(ultima_data, horizonte):
    return pd.date_range(ultima_data + pd.Timedelta(days = 1), periods = horizonte, freq = 'D', name = 'sale_date')


def eventos_futuros(datas):
    """Eventos do período previsto: somente os planejados (os não planejados valem zero)."""
    return features.criar_eventos(datas, incluir_nao_planejados = False)


class SeasonalNaive:
    """Baseline: repete as vendas da última semana conhecida."""

    nome = 'Seasonal Naive'

    def fit(self, df):
        self.ultima_semana_ = df['order_units'].iloc[-7:].values
        self.ultima_data_ = df.index.max()
        return self

    def predict(self, horizonte=config.HORIZONTE):
        datas = datas_futuras(self.ultima_data_, horizonte)
        previsao = np.tile(self.ultima_semana_, int(np.ceil(horizonte / 7)))[:horizonte]
        return pd.DataFrame({'previsao' : previsao}, index = datas)


class ModeloSARIMAX:
    """ARIMA, SARIMA ou SARIMAX ajustado no logaritmo das vendas."""

    def __init__(self, nome, order, seasonal_order=(0, 0, 0, 0), usar_eventos=False):
        self.nome = nome
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.usar_eventos = usar_eventos

    def _especificar(self):
        exog = self.dados_[config.EVENTOS] if self.usar_eventos else None
        return SARIMAX(self.dados_['log_units'], exog = exog, order = self.order, seasonal_order = self.seasonal_order)

    def fit(self, df):
        self.dados_ = df[['log_units'] + config.EVENTOS].copy()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.resultado_ = self._especificar().fit(disp = False, maxiter = 200)
        self.parametros_ = self.resultado_.params
        self.ultima_data_ = df.index.max()
        return self

    def __getstate__(self):
        # Ao salvar o modelo, guarda somente os parâmetros estimados e os dados de treino (alguns KB).
        # O objeto completo do statsmodels ocupa dezenas de MB e é reconstruído no predict.
        estado = self.__dict__.copy()
        estado.pop('resultado_', None)
        return estado

    def predict(self, horizonte=config.HORIZONTE):
        if not hasattr(self, 'resultado_'):
            self.resultado_ = self._especificar().filter(self.parametros_)    # Reaplica os parâmetros, sem reestimar
        datas = datas_futuras(self.ultima_data_, horizonte)
        exog = eventos_futuros(datas) if self.usar_eventos else None
        previsao = self.resultado_.get_forecast(horizonte, exog = exog)
        intervalo = np.exp(previsao.conf_int(alpha = 0.20).values)
        return pd.DataFrame({'previsao' : np.exp(previsao.predicted_mean.values),
                             'limite_inferior_80' : intervalo[:, 0],
                             'limite_superior_80' : intervalo[:, 1]}, index = datas)


class ModeloML:
    """XGBoost ou LightGBM com previsão direta (features de vendas passadas com defasagem >= horizonte)."""

    def __init__(self, nome, estimador, horizonte=config.HORIZONTE):
        self.nome = nome
        self.estimador = estimador
        self.horizonte = horizonte

    def fit(self, df):
        base = features.criar_features_ml(df['order_units'], df, self.horizonte).dropna()
        self.estimador.fit(base, df.loc[base.index, 'order_units'])
        self.historico_ = df['order_units'].copy()
        return self

    def predict(self, horizonte=config.HORIZONTE):
        if horizonte > self.horizonte:
            raise ValueError(f'O modelo foi treinado para prever até {self.horizonte} dias à frente.')
        datas = datas_futuras(self.historico_.index.max(), horizonte)
        vendas = pd.concat([self.historico_, pd.Series(np.nan, index = datas)])     # Vendas futuras desconhecidas
        eventos = pd.concat([features.criar_eventos(self.historico_.index), eventos_futuros(datas)])
        X = features.criar_features_ml(vendas, eventos, self.horizonte).loc[datas]
        return pd.DataFrame({'previsao' : self.estimador.predict(X)}, index = datas)


def criar_modelos(configs):
    """Retorna funções que criam uma instância nova de cada modelo candidato, a partir das configurações da etapa 4."""
    return {
        'Seasonal Naive' : lambda: SeasonalNaive(),
        'ARIMA' : lambda: ModeloSARIMAX('ARIMA', configs['arima']['order']),
        'SARIMA' : lambda: ModeloSARIMAX('SARIMA', configs['sarima']['order'], configs['sarima']['seasonal_order']),
        'SARIMAX' : lambda: ModeloSARIMAX('SARIMAX', configs['sarima']['order'], configs['sarima']['seasonal_order'], usar_eventos = True),
        'XGBoost' : lambda: ModeloML('XGBoost', xgb.XGBRegressor(**configs['xgboost'])),
        'LightGBM' : lambda: ModeloML('LightGBM', lgb.LGBMRegressor(verbose = -1, **configs['lightgbm'])),
    }
