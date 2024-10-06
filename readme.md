
# Análise de Avaliações de Cervejas

## Descrição do Projeto
Este projeto explora uma base de dados com mais de 1,5 milhão de avaliações de cervejas feitas por usuários. As avaliações abordam diversos aspectos sensoriais, como aroma, aparência, sabor e sensação, além de informações detalhadas sobre as cervejarias e os estilos de cerveja. A base é ideal para análises de tendências de consumo, preferências de estilos e insights sobre o comportamento dos consumidores de cervejas.

## Objetivos
- **Analisar a qualidade das cervejas** com base em diferentes critérios sensoriais.
- **Identificar tendências** no consumo de cervejas.
- **Construir um modelo preditivo** para estimar a nota geral da cerveja (`review_overall`) com base nas avaliações sensoriais.

## Estrutura dos Dados
A base de dados contém as seguintes variáveis:

| Variável                  | Descrição                                         |
|---------------------------|---------------------------------------------------|
| **brewery_id**            | Identificador único da cervejaria.                |
| **brewery_name**          | Nome da cervejaria.                              |
| **review_time**           | Data e hora da avaliação.                        |
| **review_overall**        | Nota geral da cerveja (escala de 1 a 5).        |
| **review_aroma**          | Avaliação do aroma (escala de 1 a 5).           |
| **review_appearance**     | Avaliação da aparência (escala de 1 a 5).       |
| **review_profilename**    | Nome de usuário do avaliador.                    |
| **beer_style**            | Estilo ou tipo de cerveja.                       |
| **review_palate**         | Avaliação da sensação na boca (escala de 1 a 5).|
| **review_taste**          | Avaliação do sabor (escala de 1 a 5).           |
| **beer_name**             | Nome da cerveja.                                 |
| **beer_abv**              | Teor alcoólico da cerveja (ABV - Alcohol by Volume). |
| **beer_beerid**           | Identificador único da cerveja.                  |

Com essa estrutura, é possível realizar uma análise detalhada da relação entre os diferentes aspectos sensoriais e as notas atribuídas pelos consumidores, proporcionando insights valiosos sobre a indústria cervejeira.

## Metodologia
1. **Importação de Bibliotecas**: Utilizamos bibliotecas como `pandas`, `numpy`, `sklearn` e `matplotlib` para manipulação de dados, modelagem e visualização.
2. **Carregamento dos Dados**: Os dados foram carregados de um arquivo CSV e armazenados em um DataFrame.
3. **Preparação dos Dados**:
   - Seleção de características e variável alvo (`review_overall`).
   - Conversão da coluna `review_time` para o formato datetime (opcional).
   - Classificação de avaliações em categorias textuais:
     - Os valores de 1 a 5 foram interpretados da seguinte forma:
       - `1.0`: "Muito ruim"
       - `1.5`: "Ruim"
       - `2.0`: "Regular"
       - `2.5`: "Razoável"
       - `3.0`: "Satisfatório"
       - `3.5`: "Agradável"
       - `4.0`: "Muito bom"
       - `4.5`: "Excelente"
       - `5.0`: "Perfeito"
4. **Divisão dos Dados**: O conjunto de dados foi dividido em conjuntos de treino e teste (80/20).
5. **Criação e Treinamento do Modelo**: Diversos modelos de regressão (Linear, Árvore de Decisão, Random Forest e K-Nearest Neighbors) foram criados e treinados para prever a nota geral da cerveja.
6. **Avaliação do Modelo**: Os modelos foram avaliados utilizando as métricas de erro quadrático médio (MSE) e o coeficiente de determinação (R²).
7. **Previsões**: Funções foram criadas para prever a nota geral com base em entradas sensoriais.
8. **Visualização dos Resultados**: Gráficos de dispersão foram gerados para comparar os valores reais e previstos, além de histogramas das avaliações.

## Resultados
Os modelos conseguiram prever a nota geral com uma precisão razoável, proporcionando insights sobre a relação entre as características sensoriais e a avaliação geral da cerveja. 

## Conclusão
Este projeto oferece uma visão abrangente sobre as avaliações de cervejas, permitindo que tanto os consumidores quanto os produtores entendam melhor as preferências e tendências no mercado.

## Autores
| Autor                             |
|-----------------------------------|
| <a href="https://github.com/Cyberfn"><img src="https://github.com/Cyberfn.png" width="200" height="200" /></a><br>**Fernando Nonato** |