def filter_range(min_value, max_value, X, y):
    """
    Filtra o dataset removendo linhas que possuem valores fora do range especificado
    
    Parâmetros:
    - min_value: valor mínimo aceitável
    - max_value: valor máximo aceitável  
    - X: DataFrame com as features
    - y: DataFrame/Series com o target
    
    Retorna:
    - X_filtrado: DataFrame X sem as linhas problemáticas
    - y_filtrado: DataFrame/Series y sem as linhas problemáticas
    """
    
    # Identificar linhas com valores fora do range
    linhas_problematicas = set()
    
    for feature in X.columns:
        mask_fora_range = (X[feature] < min_value) | (X[feature] > max_value)
        indices_fora_range = X[mask_fora_range].index.tolist()
        linhas_problematicas.update(indices_fora_range)
    
    linhas_problematicas = sorted(list(linhas_problematicas))
    
    # Remover linhas problemáticas
    X_filtrado = X.drop(index=linhas_problematicas)
    y_filtrado = y.drop(index=linhas_problematicas)
    
    # Calcular percentual de redução
    percentual_original = len(X)
    percentual_filtrado = len(X_filtrado)
    percentual_removido = len(linhas_problematicas)
    percentual_reducao = (percentual_removido / percentual_original) * 100
    
    print(f"📊 RESULTADO DA FILTRAGEM (Range: {min_value} - {max_value})")
    print(f"Dataset original: {percentual_original} observações")
    print(f"Dataset filtrado: {percentual_filtrado} observações")
    print(f"Linhas removidas: {percentual_removido} ({percentual_reducao:.2f}%)")
    print(f"Linhas mantidas: {100 - percentual_reducao:.2f}%")
    
    return X_filtrado, y_filtrado