# Tech Challenge Fase 3 | Risco de alfabetização municipal

Este projeto usa os indicadores de alfabetização de 2023 para estimar quais municípios podem ficar abaixo da meta de 2024. O resultado é um ranking que pode ajudar gestores a escolher onde investigar os indicadores com mais atenção.

O desafio é: prever se o aluno será alfabetizado. Os CSVs Gold disponíveis contêm resultados agrupados, então a implementação faz a previsão por **município e rede de ensino**. [Enunciado do desafio](docs/TechChallenge_Fase3.pdf)

## Como o projeto funciona

| Etapa | O que acontece |
| --- | --- |
| 1. Explorar | O notebook 01 verifica cobertura, valores ausentes, taxas e metas de municípios, UFs e Brasil. |
| 2. Comparar modelos | O notebook 02 usa os indicadores municipais de 2023 para prever o cumprimento da meta de 2024. Compara regressão logística, Random Forest e XGBoost e explica as previsões com SHAP. |
| 3. Avaliar e salvar | O notebook 03 avalia o Random Forest sem transformação logarítmica no teste reservado. Depois, treina com todos os registros rotulados e salva o modelo. |
| 4. Gerar ranking | O módulo `src.predizer` usa o modelo salvo para ordenar os municípios da rede coberta pela probabilidade estimada de ficar abaixo da meta no ano seguinte. |

## Dados disponíveis

| Base em `data/gold/` | Anos | Registros | Uso |
| --- | --- | ---: | --- |
| `indicadores_municipio.csv` | 2023 e 2024 | 10.271 | Análise, treinamento e previsão |
| `indicadores_uf.csv` | 2023 e 2024 | 49 | Análise exploratória |
| `indicadores_brasil.csv` | 2023 a 2025 | 3 | Análise exploratória |


## Como executar no Windows

Na raiz do projeto, com Python 3.11 ou superior, crie o ambiente e instale as dependências. O projeto foi verificado com Python 3.11.2 no ambiente local.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Gerar o ranking com o modelo salvo

```powershell
.\.venv\Scripts\python.exe -m src.predizer --ano-historico 2024 --saida resultado/ranking.csv
```

O CSV traz o código do município, o código da rede, o ano previsto (2025), a probabilidade de risco entre 0 e 1 e a classe prevista. Uma probabilidade de pelo menos 0,50 recebe classe 1. Com os dados atuais, o comando pontua 5.449 registros.

### Treinar e acompanhar a análise

Para treinar pela linha de comando e guardar uma cópia separada do modelo distribuído:

```powershell
.\.venv\Scripts\python.exe -m src.treinar --modelo tmp/documentacao/modelo.joblib
.\.venv\Scripts\python.exe -m src.predizer --ano-historico 2024 --modelo tmp/documentacao/modelo.joblib --saida tmp/documentacao/ranking.csv
```

Para acompanhar a análise no VS Code, selecione o ambiente `.venv` como kernel e execute todas as células em ordem:

1. [01 · Análise exploratória](notebooks/01_eda_analise_exploratoria.ipynb)
2. [02 · Preparação e comparação de modelos](notebooks/02_preprocessamento_modelagem.ipynb)
3. [03 · Teste reservado e modelo final](notebooks/03_modelo_producao.ipynb)

**Atenção ao arquivo salvo.** Os notebooks 02 e 03 gravam em `modelos/risco_alfabetizacao.joblib`. Execute o 03 por último para deixar salvo o modelo treinado com todos os registros rotulados.

**Ajuste para rodar o notebook 01.** Na primeira célula de código da seção “10. Comparando os anos”, remova a linha isolada `len()` antes de executar. Essa chamada exige um argumento. Mantenha as linhas que definem e contam as UFs.

## O que os resultados mostram

- Entre os 4.772 registros de município e rede com taxa oficial nos dois anos, o aumento médio foi de **2,61 pontos percentuais**. Essa média descreve o grupo com dados nos dois anos. A taxa nacional vem da tabela Brasil.
- O Random Forest ajustado sem log foi a configuração escolhida após os experimentos. No teste reservado de **923 registros**, obteve **61,43% de acurácia**, **62,00% de recall** para a classe de risco e **ROC-AUC de 0,6618**. Na prática, o modelo encontrou cerca de 6 em cada 10 casos que ficaram abaixo da meta.
- O ranking de 2025 é um cenário calculado com indicadores de 2024. O teste mediu o desempenho em municípios separados no mesmo período de 2023–2024. A Gold municipal disponível vai até 2024, por isso ainda falta confrontar o ranking com resultados municipais de 2025.

Para uma equipe de gestão, o ranking pode orientar a ordem de revisão dos indicadores e a discussão com as redes. As explicações SHAP mostram quais variáveis influenciaram a estimativa do modelo. As causas do desempenho educacional exigem investigação própria.

As evoluções mais diretamente ligadas ao desafio são obter dados individuais com rótulo de alfabetização, confirmar quando cada indicador ficou disponível, documentar a correspondência entre município e UF e validar o modelo em um ano posterior.
