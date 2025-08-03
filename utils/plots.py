import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import collections
import numpy as np

def plot_distribuicao_classes(distribuicao_classes):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    distribuicao_classes.plot(kind='bar', color=['skyblue', 'lightcoral'])
    plt.title('Distribuição das Classes (Contagem)', fontsize=14, fontweight='bold')
    plt.xlabel('Classe')
    plt.ylabel('Número de Instâncias')
    plt.xticks([0, 1], ['Olhos Fechados (0)', 'Olhos Abertos (1)'], rotation=0)

    plt.subplot(1, 2, 2)
    plt.pie(distribuicao_classes.values, labels=['Olhos Fechados (0)', 'Olhos Abertos (1)'], 
            autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])
    plt.title('Distribuição das Classes (Percentual)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()

def plotar_comparacao_feature(nome_feature, X, y, X_filtrado, y_filtrado):
    """
    Cria um plot com 4 subgráficos comparando a feature antes e depois da filtragem
    
    Parâmetros:
    - nome_feature: nome da feature a ser analisada
    - X: DataFrame original
    - y: target original
    - X_filtrado: DataFrame filtrado
    - y_filtrado: target filtrado
    """
    
    # Extrair dados da feature
    dados_original = X[nome_feature]
    dados_filtrado = X_filtrado[nome_feature]
    
    # Criar figura com 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Comparação: {nome_feature} - Original vs Filtrado', fontsize=16, fontweight='bold')
    
    # 1. Boxplot Original (superior esquerdo)
    bp1 = axes[0,0].boxplot(dados_original, patch_artist=True, widths=0.6)
    bp1['boxes'][0].set_facecolor('lightcoral')
    axes[0,0].set_title('Boxplot - Dataset Original', fontsize=14, fontweight='bold')
    axes[0,0].set_ylabel('Valor')
    axes[0,0].grid(True, alpha=0.3)
    
    # Adicionar estatísticas no boxplot original
    stats_orig = f'N: {len(dados_original)}\nMédia: {dados_original.mean():.1f}\nDP: {dados_original.std():.1f}\nMin: {dados_original.min():.1f}\nMax: {dados_original.max():.1f}'
    axes[0,0].text(0.02, 0.98, stats_orig, transform=axes[0,0].transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Distribuição Original (superior direito)
    axes[0,1].hist(dados_original, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0,1].set_title('Distribuição - Dataset Original', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Valor')
    axes[0,1].set_ylabel('Frequência')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Boxplot Filtrado (inferior esquerdo)
    bp2 = axes[1,0].boxplot(dados_filtrado, patch_artist=True, widths=0.6)
    bp2['boxes'][0].set_facecolor('lightblue')
    axes[1,0].set_title('Boxplot - Dataset Filtrado', fontsize=14, fontweight='bold')
    axes[1,0].set_ylabel('Valor')
    axes[1,0].grid(True, alpha=0.3)
    
    # Adicionar estatísticas no boxplot filtrado
    stats_filt = f'N: {len(dados_filtrado)}\nMédia: {dados_filtrado.mean():.1f}\nDP: {dados_filtrado.std():.1f}\nMin: {dados_filtrado.min():.1f}\nMax: {dados_filtrado.max():.1f}'
    axes[1,0].text(0.02, 0.98, stats_filt, transform=axes[1,0].transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Distribuição Filtrada (inferior direito)
    axes[1,1].hist(dados_filtrado, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
    axes[1,1].set_title('Distribuição - Dataset Filtrado', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('Valor')
    axes[1,1].set_ylabel('Frequência')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Imprimir resumo comparativo
    print(f"\n📈 RESUMO COMPARATIVO - {nome_feature}")
    print(f"{'='*50}")
    print(f"{'Métrica':<20} {'Original':<12} {'Filtrado':<12} {'Melhoria':<10}")
    print(f"{'='*50}")
    print(f"{'Observações':<20} {len(dados_original):<12} {len(dados_filtrado):<12} {'-':<10}")
    print(f"{'Média':<20} {dados_original.mean():<12.2f} {dados_filtrado.mean():<12.2f} {abs(dados_original.mean() - dados_filtrado.mean()):<10.2f}")
    print(f"{'Desvio Padrão':<20} {dados_original.std():<12.2f} {dados_filtrado.std():<12.2f} {dados_original.std() - dados_filtrado.std():<10.2f}")
    print(f"{'Mínimo':<20} {dados_original.min():<12.2f} {dados_filtrado.min():<12.2f} {'-':<10}")
    print(f"{'Máximo':<20} {dados_original.max():<12.2f} {dados_filtrado.max():<12.2f} {'-':<10}")
    
    # Calcular melhoria percentual no desvio padrão
    melhoria_dp = ((dados_original.std() - dados_filtrado.std()) / dados_original.std()) * 100
    print(f"{'Melhoria DP (%)':<20} {'-':<12} {'-':<12} {melhoria_dp:<10.1f}")

def plot_param_frequencies(best_params):
    # Conta quantas vezes cada valor apareceu para cada parâmetro
    param_counts = collections.defaultdict(collections.Counter)
    
    for d in best_params:
        for k, v in d.items():
            param_counts[k][v] += 1

    # Plota individualmente para cada parâmetro, variando as cores
    for param, counter in param_counts.items():
        items = sorted(counter.items(), key=lambda x: -x[1])
        labels, values = zip(*items)
        colors = sns.color_palette("Set2", len(labels))
        plt.figure(figsize=(6, 4))
        plt.bar([str(l) for l in labels], values, color=colors)
        plt.title(f'Frequência dos valores de {param}')
        plt.ylabel('Frequência')
        plt.xlabel(param)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def plot_stability_vs_metric(stds, means, stable_idxs, unstable_idxs, metric):
    plt.figure(figsize=(8, 5))
    plt.scatter([stds[i] for i in stable_idxs], [means[i] for i in stable_idxs], c='g', label='Mais estáveis')
    plt.scatter([stds[i] for i in unstable_idxs], [means[i] for i in unstable_idxs], c='r', label='Menos estáveis')
    plt.xlabel('Desvio padrão da métrica (estabilidade)')
    plt.ylabel(f'Média da métrica ({metric})')
    plt.title('Estabilidade x Média da Métrica (Separado)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_metric_per_fold(scores, metric):
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, 6), scores, marker='o')
    plt.title(f'{metric} em cada fold para o modelo mais estável')
    plt.xlabel('Fold')
    plt.ylabel(metric)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()

def plot_metric_evolution(percents, train_scores, test_scores, metric):
    
    plt.figure(figsize=(8, 5))
    plt.plot(percents * 100, train_scores, label='Treino', color='b', linestyle='-')
    plt.plot(percents * 100, test_scores, label='Teste', color='r', linestyle='--')
    plt.xlabel('% do conjunto de treino utilizado')
    plt.ylabel(f'{metric.capitalize()}')
    plt.title(f'Evolução da {metric.capitalize()} no treino e teste')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(y_test, y_pred_test, model_name):
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Matriz de Confusão (Conjunto de Teste) - Modelo {model_name}")
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.show()

def plot_roc_curve(model, X_test, y_test, model_name):
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc_score = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'Curva ROC (Conjunto de Teste) - Modelo {model_name}')
    plt.legend()
    plt.grid(True)
    plt.show()
    return auc_score

def plot_roc_curve_evolution(model, X_test, y_test, model_name):
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)
    
    y_true = y_test.values if hasattr(y_test, 'values') else y_test
    
    n_positives = np.sum(y_true == 1)
    n_negatives = np.sum(y_true == 0)
    
    tp_cumsum = np.cumsum(y_true == 1)
    fp_cumsum = np.cumsum(y_true == 0)
    
    tpr_evolution = tp_cumsum / n_positives
    fpr_evolution = fp_cumsum / n_negatives
    
    tpr_evolution = np.concatenate([[0], tpr_evolution])
    fpr_evolution = np.concatenate([[0], fpr_evolution])
    
    plt.figure(figsize=(8, 6))
    
    colors = plt.cm.plasma(np.linspace(0, 1, len(tpr_evolution)))
    
    for i in range(1, len(tpr_evolution)):
        plt.plot([fpr_evolution[i-1], fpr_evolution[i]], 
                [tpr_evolution[i-1], tpr_evolution[i]], 
                color=colors[i], alpha=0.8, linewidth=1.5)
    
    step = max(1, len(fpr_evolution)//30)  # Mostrar ~30 pontos
    scatter = plt.scatter(fpr_evolution[::step], 
                         tpr_evolution[::step], 
                         c=range(0, len(fpr_evolution), step), 
                         cmap='plasma', s=40, alpha=0.9, 
                         edgecolors='black', linewidth=0.5)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Linha de referência')
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.title(f'Evolução Temporal da Curva ROC - {model_name}')
    plt.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Ordem Temporal das Amostras', rotation=270, labelpad=20)
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_main_metrics(metrics_dict, model_name):
    labels = [
        "Acurácia (Treino)", "Acurácia (Teste)",
        "F1 (Treino)", "F1 (Teste)",
        "Precisão (Treino)", "Precisão (Teste)",
        "Recall (Treino)", "Recall (Teste)"
    ]
    metrics = [
        metrics_dict["accuracy_train"], metrics_dict["accuracy_test"],
        metrics_dict["f1_train"], metrics_dict["f1_test"],
        metrics_dict["precision_train"], metrics_dict["precision_test"],
        metrics_dict["recall_train"], metrics_dict["recall_test"]
    ]
    colors = sns.color_palette("Paired", 8)
    plt.figure(figsize=(8, 4))
    bars = plt.bar(range(len(metrics)), metrics, color=colors)
    plt.xticks(range(len(metrics)), labels, rotation=25, ha='right')
    plt.title(f'Métricas principais - Modelo {model_name}')
    y_min = max(min(metrics) - 0.1, 0)  # considera o menor valor de todas as métricas
    plt.ylim(y_min, 1)
    plt.tight_layout(pad=1)
    plt.show()