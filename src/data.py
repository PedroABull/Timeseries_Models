"""Construção da série de vendas diárias a partir dos arquivos originais da Olist (etapas 1 e 2)."""

import logging

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def carregar_vendas(pasta_dados=config.PASTA_DADOS):
    """Monta a base de vendas: uma linha por produto em cada pedido entregue (etapa 1 + limpeza da etapa 2)."""
    itens = pd.read_csv(pasta_dados / 'olist_order_items_dataset.csv')
    pedidos = pd.read_csv(pasta_dados / 'olist_orders_dataset.csv')
    produtos = pd.read_csv(pasta_dados / 'olist_products_dataset.csv')

    # Unidades de cada produto em cada pedido
    vendas = (itens.groupby(['order_id', 'product_id'], as_index = False)
                   .agg(order_units = ('order_item_id', 'count')))

    # Somente pedidos entregues, com a data da compra como data da venda
    pedidos = pedidos.loc[pedidos['order_status'] == 'delivered', ['order_id', 'order_purchase_timestamp']]
    vendas = vendas.merge(pedidos, on = 'order_id', how = 'inner')

    # Excluindo os produtos sem categoria (mesmo tratamento da etapa 2)
    vendas = vendas.merge(produtos[['product_id', 'product_category_name']], on = 'product_id', how = 'left')
    vendas = vendas.dropna(subset = ['product_category_name'])

    vendas['sale_date'] = pd.to_datetime(vendas['order_purchase_timestamp']).dt.normalize()
    return vendas[['sale_date', 'order_units']]


def construir_serie_diaria(pasta_dados=config.PASTA_DADOS, inicio=config.INICIO_SERIE, fim=config.FIM_SERIE):
    """Retorna a série contínua de unidades vendidas por dia no período útil."""
    vendas = carregar_vendas(pasta_dados)
    serie = vendas.groupby('sale_date')[['order_units']].sum().asfreq('D', fill_value = 0)
    serie = serie.loc[inicio:fim]
    serie.index.name = 'sale_date'
    logger.info('Série diária construída: %s a %s (%d dias)', serie.index.min().date(), serie.index.max().date(), len(serie))
    return serie
