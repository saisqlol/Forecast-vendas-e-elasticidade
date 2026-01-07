# Modelo de Previsão de Venda Média Diária (VMD)

Este diretório contém a lógica para calcular a Venda Média Diária (VMD) realizada e prever o VMD futuro para SKUs específicos. O processo é dividido em duas etapas principais: **treinamento** e **previsão**.

## Estrutura e Fluxo de Trabalho

O sistema é projetado para ser eficiente, separando o processo computacionalmente intensivo de treinamento do processo rápido de previsão diária.

### 1. Treinamento dos Modelos (`train_models.py`)

-   **O que faz?** Este script é responsável por construir os modelos preditivos. Para cada SKU em uma lista, ele:
    1.  Busca o histórico de vendas completo dos últimos 2 anos no BigQuery.
    2.  Usa a biblioteca `pmdarima` (`auto_arima`) para encontrar os melhores parâmetros (p,d,q)(P,D,Q,s) para um modelo SARIMAX.
    3.  Treina o modelo SARIMAX com os dados históricos e os melhores parâmetros encontrados.
    4.  Salva o objeto do modelo treinado (`.pkl`) e um arquivo com os parâmetros (`.json`) na pasta `Modelos_VMD`.
-   **Quando usar?** Execute este script esporadicamente (ex: uma vez por mês ou a cada trimestre) para garantir que os modelos estejam atualizados com os padrões de venda mais recentes.

### 2. Geração de Previsões (`predict_vmd.py`)

-   **O que faz?** Este script é leve e rápido, projetado para ser executado diariamente. Para cada SKU:
    1.  Carrega o modelo pré-treinado (`.pkl`) e os parâmetros (`.json`) da pasta `Modelos_VMD`.
    2.  Busca o histórico de vendas recente no BigQuery.
    3.  Calcula o VMD realizado dos últimos 60 dias.
    4.  Usa o modelo carregado para prever a demanda para os próximos 7 dias.
    5.  Calcula o VMD previsto para amanhã e para a próxima semana.
    6.  Retorna um DataFrame com os resultados.
-   **Quando usar?** Execute este script diariamente para obter as previsões de VMD mais recentes.

---

## Exemplo de Cálculo: SKU 88264

Vamos usar um exemplo prático para ilustrar como os cálculos são feitos, assumindo os seguintes dados para o SKU **88264**:

-   **Demanda Total nos Últimos 60 Dias:** 3.372 unidades.

### VMD Realizado (Histórico)

Esta é a média simples da demanda durante o período. A fórmula é:

$$
\text{VMD}_{\text{Realizado 60D}} = \frac{\sum_{i=1}^{60} \text{Demanda}_{\text{dia } i}}{60}
$$

Para o nosso SKU:

$$
\text{VMD}_{\text{Realizado 60D}} = \frac{3372}{60} = 56.2
$$

### VMD Previsto para Amanhã

Para esta previsão, removemos a demanda do dia mais antigo (dia 60) e adicionamos a previsão de demanda do modelo para amanhã.

$$
\text{VMD}_{\text{Previsto Amanhã}} = \frac{\left( \sum_{i=1}^{59} \text{Demanda}_{\text{dia } i} \right) + \text{Previsão}_{\text{Amanhã}}}{60}
$$

**Exemplo numérico:**
- Suponha que a demanda do dia mais antigo (há 60 dias) foi de **40 unidades**.
- A soma dos últimos 59 dias é: `3372 - 40 = 3332`.
- Suponha que o modelo preveja uma demanda de **58 unidades** para amanhã.

O cálculo será:

$$
\text{VMD}_{\text{Previsto Amanhã}} = \frac{3332 + 58}{60} = \frac{3390}{60} = 56.5
$$

### VMD Previsto para a Próxima Semana

A lógica é semelhante. Removemos os 7 dias mais antigos da janela e adicionamos as 7 previsões futuras geradas pelo modelo.

$$
\text{VMD}_{\text{Previsto 7D}} = \frac{\left( \sum_{i=1}^{53} \text{Demanda}_{\text{dia } i} \right) + \left( \sum_{j=1}^{7} \text{Previsão}_{\text{dia } j} \right)}{60}
$$

**Exemplo numérico:**
- Suponha que a soma da demanda dos 7 dias mais antigos foi de **350 unidades**.
- A soma dos últimos 53 dias é: `3372 - 350 = 3022`.
- Suponha que a soma das previsões do modelo para os próximos 7 dias seja de **400 unidades**.

O cálculo será:

$$
\text{VMD}_{\text{Previsto 7D}} = \frac{3022 + 400}{60} = \frac{3422}{60} \approx 57.03
$$
