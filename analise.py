# Importando Bibliotecas Necessárias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor  
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsRegressor  
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Carregando os Dados
df = pd.read_csv('BaseCervejas.csv')
df2 = df.copy()

# Criando Histograma das Avaliações
plt.figure(figsize=(10, 6))
plt.hist(df2['review_overall'], bins=30, color='blue', alpha=0.7, edgecolor='black')
plt.title('Distribuição das Notas Gerais das Cervejas')
plt.xlabel('Nota Geral (review_overall)')
plt.ylabel('Frequência')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Gráfico de dispersão - Relação entre teor alcoólico (ABV) e nota geral
plt.figure(figsize=(10,6))
sns.scatterplot(x='beer_abv', y='review_overall', data=df2, alpha=0.5)
plt.title('Relação entre Teor Alcoólico (ABV) e Nota Geral')
plt.xlabel('Teor Alcoólico (%)')
plt.ylabel('Nota Geral')
plt.show()

# Gráfico de barras - Média das avaliações por estilo de cerveja
plt.figure(figsize=(12,8))
avg_review_by_style = df2.groupby('beer_style')['review_overall'].mean().sort_values(ascending=False).head(10)
avg_review_by_style.plot(kind='barh', color='skyblue')
plt.title('Média das Notas Gerais por Estilo de Cerveja (Top 10)')
plt.xlabel('Nota Média')
plt.ylabel('Estilo de Cerveja')
plt.show()

# Heatmap - Correlação entre as avaliações
corr = df2[['review_aroma', 'review_appearance', 'review_palate', 'review_taste', 'review_overall']].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlação entre as Avaliações')
plt.show()

# Selecionando Características e Alvo
features = df2[['review_taste']]
target = df2['review_overall']

# Convertendo para datetime
df2['review_time'] = pd.to_datetime(df2['review_time'], unit='s')

# Dividindo os Dados
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Modelos de Regressão

# ---------------------------- Regressão Linear --------------------------------
# Instanciando o modelo de Regressão Linear
linear_model = LinearRegression()

# Treinando o modelo com os dados de treino
linear_model.fit(X_train, y_train)

# Fazendo previsões nos dados de teste
linear_predictions = linear_model.predict(X_test)

# Calculando o erro quadrático médio (MSE) entre os valores reais e previstos
linear_mse = mean_squared_error(y_test, linear_predictions)

# Calculando o coeficiente de determinação (R²) para avaliar o ajuste do modelo
linear_r2 = r2_score(y_test, linear_predictions)

# Calculando a acurácia da regressão linear
# A acurácia é definida como a proporção de previsões cuja diferença em relação ao valor real é menor que 10%
linear_accuracy = np.mean(np.abs((y_test - linear_predictions) / y_test) < 0.1)  # Acurácia entre 0 e 1

# Exibindo as métricas de desempenho: MSE, R² e acurácia
print(f'Regressão Linear - MSE: {linear_mse:.2f}, R²: {linear_r2:.2f}, Acurácia: {linear_accuracy:.2f}')
# ---------------------------- Regressão Logística ------------------------------
# Convertendo a variável alvo para binária
# Classifica como 1 se o valor for maior que 3.0 e 0 caso contrário
target_binary = (target > 3.0).astype(int)

# Dividindo os dados em conjuntos de treino e teste
# Utiliza 80% dos dados para treino e 20% para teste
X_train_logistic, X_test_logistic, y_train_logistic, y_test_logistic = train_test_split(features, target_binary, test_size=0.2, random_state=42)

# Instanciando o modelo de Regressão Logística
logistic_model = LogisticRegression()

# Treinando o modelo com os dados de treino
logistic_model.fit(X_train_logistic, y_train_logistic)

# Fazendo previsões nos dados de teste
logistic_predictions = logistic_model.predict(X_test_logistic)

# Calculando o erro quadrático médio (MSE) das previsões
# MSE é usado apenas como referência aqui, mesmo que não seja a métrica mais apropriada para classificação
logistic_mse = mean_squared_error(y_test_logistic, logistic_predictions)

# Calculando a acurácia, que é a proporção de previsões corretas
logistic_accuracy = accuracy_score(y_test_logistic, logistic_predictions)

# Exibindo as métricas de desempenho: MSE e acurácia
print(f'Regressão Logística - MSE: {logistic_mse:.2f}, Acurácia: {logistic_accuracy:.2f}')
# ---------------------------- Árvore de Decisão --------------------------------
# Instanciando o modelo de Regressão com Árvore de Decisão
decision_tree_model = DecisionTreeRegressor(random_state=42)

# Treinando o modelo de Árvore de Decisão com os dados de treino
decision_tree_model.fit(X_train, y_train)

# Fazendo previsões nos dados de teste
decision_tree_predictions = decision_tree_model.predict(X_test)

# Calculando o erro quadrático médio (MSE) entre os valores reais e previstos
decision_tree_mse = mean_squared_error(y_test, decision_tree_predictions)

# Calculando o coeficiente de determinação (R²) para avaliar o ajuste do modelo
decision_tree_r2 = r2_score(y_test, decision_tree_predictions)

# Calculando a acurácia da Árvore de Decisão
# A acurácia é a proporção de previsões cuja diferença em relação ao valor real é menor que 10%
decision_tree_accuracy = np.mean(np.abs((y_test - decision_tree_predictions) / y_test) < 0.1)

# Exibindo as métricas de desempenho: MSE, R² e acurácia
print(f'Árvore de Decisão - MSE: {decision_tree_mse:.2f}, R²: {decision_tree_r2:.2f}, Acurácia: {decision_tree_accuracy:.2f}')
# ---------------------------- Random Forest -------------------------------------
# Instanciando o modelo de Regressão com Random Forest
random_forest_model = RandomForestRegressor(random_state=42)

# Treinando o modelo Random Forest com os dados de treino
random_forest_model.fit(X_train, y_train)

# Fazendo previsões nos dados de teste
random_forest_predictions = random_forest_model.predict(X_test)

# Calculando o erro quadrático médio (MSE) entre os valores reais e previstos
random_forest_mse = mean_squared_error(y_test, random_forest_predictions)

# Calculando o coeficiente de determinação (R²) para avaliar o ajuste do modelo
random_forest_r2 = r2_score(y_test, random_forest_predictions)

# Calculando a acurácia do modelo Random Forest
# A acurácia é a proporção de previsões cuja diferença em relação ao valor real é menor que 10%
random_forest_accuracy = np.mean(np.abs((y_test - random_forest_predictions) / y_test) < 0.1)

# Exibindo as métricas de desempenho: MSE, R² e acurácia
print(f'Random Forest - MSE: {random_forest_mse:.2f}, R²: {random_forest_r2:.2f}, Acurácia: {random_forest_accuracy:.2f}')
# ---------------------------- K-Nearest Neighbors(KNN) ----------------------------
# Instanciando o modelo de Regressão K-Nearest Neighbors (KNN)
# Definindo o número de vizinhos a considerar nas previsões (n_neighbors=5)
knn_model = KNeighborsRegressor(n_neighbors=5)

# Treinando o modelo KNN com os dados de treino
knn_model.fit(X_train, y_train)

# Fazendo previsões nos dados de teste
knn_predictions = knn_model.predict(X_test)

# Calculando o erro quadrático médio (MSE) entre os valores reais e previstos
knn_mse = mean_squared_error(y_test, knn_predictions)

# Calculando o coeficiente de determinação (R²) para avaliar o ajuste do modelo
knn_r2 = r2_score(y_test, knn_predictions)

# Calculando a acurácia do modelo KNN
# A acurácia é a proporção de previsões cuja diferença em relação ao valor real é menor que 10%
knn_accuracy = np.mean(np.abs((y_test - knn_predictions) / y_test) < 0.1)

# Exibindo as métricas de desempenho: MSE, R² e acurácia
print(f'K-Nearest Neighbors - MSE: {knn_mse:.2f}, R²: {knn_r2:.2f}, Acurácia: {knn_accuracy:.2f}')
# ----------- Support Vector Classifier (SVC)--------------------------------
# Definindo o modelo de Support Vector Classifier (SVC)
# svc_model = SVC()

# Treinando o modelo SVC com os dados de treino binários
# O modelo é ajustado utilizando os dados de entrada (X_train_logistic) e a variável alvo binária (y_train_logistic)
# svc_model.fit(X_train_logistic, y_train_logistic)

# Fazendo previsões nos dados de teste
# svc_predictions = svc_model.predict(X_test_logistic)

# Calculando a acurácia para o modelo SVC
# A acurácia é a proporção de previsões corretas em relação aos valores reais da variável alvo binária
# svc_accuracy = accuracy_score(y_test_logistic, svc_predictions)

# Exibindo a acurácia do modelo SVC
# print(f'Support Vector Classifier - Acurácia: {svc_accuracy:.2f}')
# ---------------------------- Previsões -----------------------------------------

def predict_overall_review_linear(taste):
    input_data = pd.DataFrame({
        'review_taste': [taste]
    })
    prediction = linear_model.predict(input_data)
    return prediction[0]  # Retornando o valor previsto sem arredondar

def predict_overall_review_logistic(review_taste):
    review_taste_df = pd.DataFrame({'review_taste': [review_taste]})
    prediction = logistic_model.predict(review_taste_df)
    return "Boa Avaliação" if prediction[0] == 1 else "Má Avaliação"

def predict_overall_review_decision_tree(taste):
    input_data = pd.DataFrame({
        'review_taste': [taste]
    })
    prediction = decision_tree_model.predict(input_data)
    return prediction[0]  # Retornando o valor previsto sem arredondar

def predict_overall_review_random_forest(taste):
    input_data = pd.DataFrame({
        'review_taste': [taste]
    })
    prediction = random_forest_model.predict(input_data)
    return prediction[0]  # Retornando o valor previsto sem arredondar

def predict_overall_review_knn(taste):
    input_data = pd.DataFrame({
        'review_taste': [taste]
    })
    prediction = knn_model.predict(input_data)
    return "Boa Avaliação" if prediction[0] == 1 else "Má Avaliação"

# ----------------- Calculando a média do sabor (review_taste) -----------------
taste_mean = df2['review_taste'].mean()
print(f"\n Média do sabor (review_taste): {taste_mean:.1f} \n")

# ---------------------------- Previsões com Regressão Linear -------------------
predicted_overall_linear = predict_overall_review_linear(taste_mean)
print(f"Regressão Linear -> Nota Geral Prevista: {predicted_overall_linear:.1f}")

# ---------------------------- Previsões com Regressão Logística ----------------
predicted_class_logistic = predict_overall_review_logistic(taste_mean)
print(f"Regressão Logística -> Classificação Prevista: {predicted_class_logistic}")

# ---------------------------- Previsões com Árvore de Decisão -------------------
predicted_overall_tree = predict_overall_review_decision_tree(taste_mean)
print(f"Árvore de Decisão -> Nota Geral Prevista: {predicted_overall_tree:.1f}")

# ---------------------------- Previsões com Random Forest -----------------------
predicted_overall_random_forest = predict_overall_review_random_forest(taste_mean)
print(f"Random Forest -> Nota Geral Prevista: {predicted_overall_random_forest:.1f}")

# ---------------------------- Previsões com KNN ---------------------------------
predicted_class_knn = predict_overall_review_knn(taste_mean)
print(f"KNN -> Classificação Prevista: {predicted_class_knn}")

# -------------- Avaliando Modelos com Matriz de Confusão e Previsões --------------
def plot_confusion_matrix(y_true, y_pred, title='Matriz de Confusão'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.show()

#  ----------------Plotando matriz de confusão para o modelo de Regressão Logística ----------------
plot_confusion_matrix(y_test_logistic, logistic_predictions, title='Matriz de Confusão - Regressão Logística')