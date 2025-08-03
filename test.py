import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo 
import utils.plots as plots
from utils import filtrar_por_range


import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 51

eeg_eye_state = fetch_ucirepo(id=264) 

X = eeg_eye_state.data.features 
y = eeg_eye_state.data.targets

X, y = filtrar_por_range(3000, 6000, X, y)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# 2. Normalizar apenas as features (X)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from tqdm.auto import tqdm  # Importando tqdm para as barras de progresso

def busca_hiperparametros(estimator, params, metric='accuracy'):
    """
    Realiza 20 iterações do RandomizedSearchCV com random state variando a cada iteração.
    Armazena as métricas e retorna os melhores parâmetros.
    """
    resultados = []
    melhores_parametros = []
    melhores_modelos = []
    
    for i in tqdm(range(20), desc="Buscando Hiperparâmetros"):  # Barra de progresso para 20 iterações
        randomized_search = RandomizedSearchCV(
            estimator,
            param_distributions=params,
            n_iter=10,
            scoring=metric,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + i),
            random_state=RANDOM_STATE + i,
            n_jobs=-1
        )
        randomized_search.fit(X_train, y_train)
        resultados.append(randomized_search.cv_results_)
        melhores_parametros.append(randomized_search.best_params_)
        melhores_modelos.append(randomized_search.best_estimator_)
    
    resultados_finais = pd.DataFrame(resultados[0])
    for i in range(1, len(resultados)):
        resultados_finais = pd.concat([resultados_finais, pd.DataFrame(resultados[i])], ignore_index=True)
    
    parametros_frequentes = pd.Series([str(params) for params in melhores_parametros]).value_counts().head(10)
    
    colunas_numericas = resultados_finais.select_dtypes(include=[np.number]).columns
    medias = resultados_finais[colunas_numericas].mean(axis=0)
    desvio_padrao = resultados_finais[colunas_numericas].std(axis=0)

    return medias, desvio_padrao, parametros_frequentes, melhores_modelos

from sklearn.model_selection import StratifiedKFold


def treinamento_progressivo(best_model):
    acuracia_treino = []
    precisao_treino = []
    recall_treino = []
    f1_treino = []
    acuracia_teste = []
    precisao_teste = []
    recall_teste = []
    f1_teste = []
    
    for i in tqdm(range(1, 21), desc="Treinamento Progressivo"):  # Barra de progresso para o treinamento progressivo
        X_treino_parcial = X_train[:int(0.05 * i * len(X_train))]
        y_treino_parcial = y_train[:int(0.05 * i * len(y_train))]
        
        best_model.fit(X_treino_parcial, y_treino_parcial)
        
        y_pred_treino = best_model.predict(X_treino_parcial)
        y_pred_teste = best_model.predict(X_test)
        
        acuracia_treino.append(accuracy_score(y_treino_parcial, y_pred_treino))
        precisao_treino.append(precision_score(y_treino_parcial, y_pred_treino))
        recall_treino.append(recall_score(y_treino_parcial, y_pred_treino))
        f1_treino.append(f1_score(y_treino_parcial, y_pred_treino))
        
        acuracia_teste.append(accuracy_score(y_test, y_pred_teste))
        precisao_teste.append(precision_score(y_test, y_pred_teste))
        recall_teste.append(recall_score(y_test, y_pred_teste))
        f1_teste.append(f1_score(y_test, y_pred_teste))
    
    plots.plot_progressive_training_progress(
        acuracia_treino, precisao_treino, recall_treino, f1_treino,
        acuracia_teste, precisao_teste, recall_teste, f1_teste
    )
    
    plots.plot_metrics_comparison(best_model, X_test, y_test)

from sklearn.tree import DecisionTreeClassifier
        
params_dt = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

medias, desvio_padrao, parametros_frequentes, resultados = busca_hiperparametros(DecisionTreeClassifier(), params_dt, metric='accuracy')

from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from tqdm.auto import tqdm
import numpy as np

def escolher_melhor_modelo(melhores_modelos, X_train, y_train, metric='accuracy'):
    """
    Realiza a segunda validação cruzada k-fold (5) nos 20 melhores modelos para escolher o melhor modelo
    baseado na estabilidade e métrica principal.
    """
    melhor_modelo = None
    melhor_media = -np.inf
    menor_desvio = np.inf
    
    # Verifique os valores únicos em y_train
    print(f"Valores únicos em y_train: {np.unique(y_train)}")

    # Verifica se o problema é de classificação ou regressão
    if len(np.unique(y_train)) < 20:  # Para classificação binária (0 ou 1)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    else:  # Para regressão (target contínuo)
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)  # Ou ShuffleSplit

    # Validação cruzada nos 20 melhores modelos
    for modelo in tqdm(melhores_modelos, desc="Validando Melhores Modelos"):  # Barra de progresso para validação dos melhores modelos
        scores = cross_val_score(modelo, X_train, y_train, cv=cv, scoring=metric)
        media_score = np.mean(scores)
        desvio_score = np.std(scores)
        
        # Escolher o modelo com melhor média e menor desvio padrão
        if media_score > melhor_media or (media_score == melhor_media and desvio_score < menor_desvio):
            melhor_modelo = modelo
            melhor_media = media_score
            menor_desvio = desvio_score
    
    return melhor_modelo

print("Valores únicos em y_train antes da função:", np.unique(y_train))


melhor_modelo = escolher_melhor_modelo(resultados, X_train, y_train , metric='accuracy')

# Treinamento progressivo do melhor modelo
treinamento_progressivo(melhor_modelo)