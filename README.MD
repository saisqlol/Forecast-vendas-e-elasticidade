## Objetivo do projeto

Este projeto tem como visão, desenvolver uma ferramenta de previsão de vendas utilizando como variáveis exógenas / independentes o preço e dummies de dia da semana. O objetivo é encontrar o melhor modelo para cada produto (SKU) e usar seus coeficientes para prever a demanda futura.

A previsão é feita comparando dois modelos principais:
1.  **Validação Cruzada em Séries Temporais (TSCV):** Um modelo de regressão robusto que respeita a ordem cronológica dos dados.
2.  **ARIMAX (AutoRegressive Integrated Moving Average with eXogenous variables):** Um modelo de série temporal que prevê valores futuros com base em seus próprios valores passados e em variáveis externas.
    *   **SARIMAX:** Uma versão mais complexa que inclui componentes sazonais, mas que foi substituída pelo ARIMAX neste projeto para garantir maior estabilidade nas previsões.

### Premissas

Séries Estacionárias (A base para previsões confiáveis)

Em termos de dados, uma série estacionária é aquela cujas propriedades estatísticas (como média e variância) não mudam ao longo do tempo. Ela não tem tendência de crescimento ou queda.

Por que isso é importante em modelos de séries temporais?

Modelos como o ARIMAX funcionam muito melhor com dados "comportados" (estacionários). O objetivo deles é capturar o relacionamento entre um ponto no tempo e os pontos anteriores. Se a série tem uma forte tendência de crescimento, o modelo fica "confuso".

No projeto, a função `encontrar_melhores_parametros_sarimax` força a criação de um modelo estacionário (`d=0`), transformando-o em um modelo de **Regressão com Erros ARMA**. Isso significa que a "âncora" da previsão vem das variáveis que entendemos (preço, dias da semana), e os componentes de série temporal (AR e MA) apenas ajustam os "erros" ou padrões residuais, garantindo previsões muito mais estáveis.

### Modelo ARIMAX (Anteriormente SARIMAX)

O nome pode ser dividido em **ARIMA** (a parte que olha para o tempo) e **X** (a parte que olha para fatores externos). A parte Sazonal (S) foi removida intencionalmente.

**Por que a mudança de SARIMAX para ARIMAX?**

Descobrimos que o componente Sazonal (`S`) do SARIMAX, quando combinado com as dummies de dia da semana, criava um modelo excessivamente complexo e instável para previsões futuras. Ao remover a busca por sazonalidade interna (`seasonal=False`), forçamos o modelo a se basear exclusivamente nas variáveis externas que fornecemos (preço e dias da semana), resultando em previsões muito mais robustas e intuitivas.

**ARIMA (p, d, q)**

*   **AR (Autoregressivo) - `p`:** O modelo usa valores passados da própria variável para prever o futuro.
*   **I (Integrado) - `d`:** No nosso projeto, forçamos `d=0` para garantir a estabilidade. A estacionariedade é garantida pela estrutura de regressão.
*   **MA (Média Móvel) - `q`:** O modelo olha para os erros de previsão passados para se corrigir.

### Validação Cruzada em séries temporais

Validação Cruzada para Séries Temporais respeita a ordem do tempo. O modelo sempre é treinado com dados do passado para prever o futuro, simulando como ele vai operar na vida real.

O projeto (`FNC_TSCV.py`) usa uma técnica chamada `TimeSeriesSplit`, que faz exatamente isso. Para `n_splits=10`, ele executa 10 simulações de previsão, expandindo a janela de treino a cada passo e usando o "próximo" pedaço de dados para teste. Ao final, a média dos erros dessas 10 simulações nos dá uma medida realista da performance do modelo.

### Aplicação e Pipeline

O fluxo de trabalho é dividido em duas etapas principais: **Treino** e **Teste/Previsão**.

#### 1. Treinamento (`Calculadora_Elasticidade.ipynb` - Célula "Treino do modelo")

- **Filtragem:** A função `lista_produtos` seleciona os SKUs a serem processados com base nos filtros definidos (Classificação, Ativo, ou uma lista específica de SKUs).
- **Loop de Treinamento:** Para cada SKU, o `pipeline_completa_skus` executa os seguintes passos:
    - **Coleta de Dados:** A função `Base_venda` busca o histórico de vendas.
    - **Treino do TSCV:** O modelo de validação cruzada é treinado.
    - **Treino do ARIMAX:** A função `encontrar_melhores_parametros_sarimax` utiliza a biblioteca **`pmdarima`** para realizar uma busca automática (`auto_arima`) pela melhor configuração de parâmetros (p, q). Ela é configurada para construir um modelo estável (com `d=0` e `seasonal=False`) e depois o treina.
    - **Geração de Relatório:** A função `gerar_relatorio_comparacao` é chamada para consolidar os resultados.
- **Seleção do Modelo Ideal:** Dentro da `gerar_relatorio_comparacao`, é criada a coluna `modelo_ideal` com base em duas regras:
    1.  **Critério de Segurança (RMSE):** Se o erro (RMSE) do TSCV for maior que o do ARIMAX, o **ARIMAX é o escolhido**.
    2.  **Critério Principal (AIC):** Caso contrário, o modelo com o **menor AIC** é selecionado.
- **Salvamento:** Os objetos dos modelos treinados são salvos na pasta `Modelos`, e um `Relatorio_Modelos_Consolidado.csv` com os resultados de todos os SKUs é salvo na pasta `Resultados`.

#### 2. Teste e Previsão (`Calculadora_Elasticidade.ipynb` - Célula "Teste do modelo")

- **Leitura de Cenários:** A célula lê o arquivo `Precos_para_previsoes.xlsx` para identificar para quais SKUs e com quais preços as previsões devem ser geradas.
- **Loop de Previsão:** Para cada SKU encontrado no arquivo de preços:
    - **Carregamento:** Os modelos TSCV e ARIMAX correspondentes são carregados da pasta `Modelos`.
    - **Previsão de 30 Dias:** A função `pred_prox_30_dias` é chamada para gerar a previsão futura, que agora inclui a coluna `previsao_total` baseada na mesma lógica do `modelo_ideal`.

### Resultados

Os resultados são visualizados através dos gráficos gerados e dos DataFrames consolidados, permitindo uma análise completa da performance e das projeções de cada modelo.


