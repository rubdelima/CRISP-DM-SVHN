# CRISP-DM Phase 1 - Business Understanding

# 1 - Determinando Objetivos de Negócio

Neste primeiro tópico, nosso objetivo é compreender o contexto de negócio no qual o conjunto de dados SVHN_medium está inserido, identificando oportunidades de melhoria por meio da aplicação do modelo CRISP-DM nesta base de dados. Para isso, exploraremos como os dados podem ser utilizados para otimizar processos, impulsionar a tomada de decisões estratégicas e avaliar modelos preditivos.

## 1.1 - Background

A identificação de números de casas é uma tarefa essencial para diversas aplicações na sociedade moderna, principalmente em setores como segurança, transporte, mapeamento urbano e serviços de emergência. O SVHN_medium é um dataset reduzido pela metade dos SVHN original, que fornece imagens de números de casas retiradas de fotos do Google Street View, representando um desafio para modelos de visão computacional, especialmente quando se trata de reconhecer e classificar números em ambientes urbanos dinâmicos.

Nesse contexto, empresas e governos podem utilizar modelos de aprendizado de máquina para automatizar o reconhecimento de endereços, ajudando na otimização de serviços urbanos como entrega de mercadorias, serviços de segurança e localização de endereços para atendimento a emergências. A eficiência na identificação de números de casas pode transformar a forma como as empresas prestam serviços, garantindo rapidez, precisão e redução de erros humanos.

Além disso, a tecnologia de reconhecimento de números de casas é crucial para a implementação de sistemas de navegação inteligente. Aplicações como GPS e sistemas de mapas precisam ser capazes de identificar corretamente os números de casas em tempo real, oferecendo direções precisas e eficientes. Esses sistemas dependem de modelos de aprendizado profundo para interpretar e classificar as imagens capturadas, como as do SVHN_medium, de maneira rápida e robusta, garantindo uma experiência de usuário sem falhas.

A precisão na classificação dos números de casas também é importante para a criação de bases de dados de endereços, utilizados por governos e empresas para formularem políticas públicas, realizarem pesquisas demográficas e monitorarem o crescimento urbano. No setor de segurança, o uso de modelos preditivos de reconhecimento de endereços pode aprimorar a vigilância em tempo real, proporcionando identificações rápidas de localizações, fundamentais para intervenções rápidas em situações de emergência. 

Portanto, a utilização do **SVHN_medium** é um ponto de partida importante para o avanço da automação na identificação de endereços, proporcionando um diferencial competitivo para empresas de tecnologia e serviços públicos. A implementação de sistemas robustos de reconhecimento de números de casas pode transformar significativamente a forma como os serviços urbanos são prestados e otimizados, promovendo maior segurança, eficiência e inovação.

## 1.2 - Objetivos de Negócio

O objetivo é criar modelos de aprendizado de máquina que possam reconhecer, de forma precisa e robusta, números de casas em imagens de ruas e edifícios. Com isso, espera-se fornecer uma ferramenta eficaz para sistemas automatizados de leitura de placas e endereços, beneficiando aplicações como mapeamento urbano, sistemas de navegação e soluções de segurança. A meta é alcançar uma maior taxa de acerto nas previsões, reduzindo o erro em contextos reais, como na automação de reconhecimento de números de casas, o que impacta positivamente a melhoria dos serviços urbanos e a otimização do trabalho de sistemas de inteligência artificial aplicados à visão computacional.

## 1.3 - **Critérios de Sucesso do Objetivo de Negócio**

O sucesso do projeto será medido pela melhoria do F1-score, precisão, recall e ROC-AUC nas tarefas de classificação de números de casas presentes nas imagens do dataset SVHN_medium. A escolha dessas métricas se justifica pela natureza do problema, onde a acurácia por si só pode ser insuficiente, especialmente em cenários com variação nos padrões visuais das imagens e com dados de diferentes fontes. O F1-score equilibra precisão e recall, assegurando que os modelos consigam identificar corretamente os números de casas mesmo em casos difíceis ou com distorções nas imagens. O ROC-AUC, por sua vez, será utilizado para avaliar a capacidade do modelo em distinguir entre as diversas classes (dígitos), mesmo quando as distribuições das classes são desiguais.

---

# 2 - Avaliação da Situação

## 2.1 - Inventário de Recursos

O projeto possui como base de dados o SVHN_medium, um subconjunto reduzido da base de dados SVHN (Street View House Numbers), que fornece imagens de números de casas retiradas de fotos do Google Street View. O conjunto SVHN_medium mantém a estrutura visual do dataset original, mas ele possui uma amostragem reduzida visando faciltar a execução em ambientes com menor capacidade computacional. Para desenvolver este projeto, teremos como recursos disponíveis:

- Base de dados SVHN_medium, composta por imagens rotuladas e prontas para tarefas de classificação supervisionada
- Ambientes de execução, como Google Colab ou o Jupyter Notebook, que possibilitam o uso de recursos computacionais acessíveis
- Ferramentas e bibliotecas de ciência de dados e visão computacional, como scikit-learn e Matplotlib
- GitHub, utilizado como ambiente de versionamento de código, registro de experimentos e colaboração entre os integrantes do grupo

## 2.2 - **Requisitos, Suposições e Restrições**

Assume-se que a base de dados é suficientemente diversificada, permitindo o reconhecimento de todos os digitos, e que os rótulos associadas a estes são verdadeiros. 

Como principal requisito, está presente o reconhecimento de números em quaisquer situações — sob diferentes iluminações ou background, por exemplo — dada a restrição de que esses números estão bem delimitados na imagem, ocupando posição central e principal na fotografia, visto que foi esse o estilo das imagens escolhidas para representar o dataset.

## 2.3 - **Riscos e Contingências**

Embora a tarefa de reconhecimento de caracteres seja um problema que já foi, de maneira geral, resolvido, o seu reconhecimento em contexto — ou seja, como parte de imagens complexas — é mais difícil. Diversos problemas, como imagens com baixa resolução, sem foco, ou borradas, são exemplos de desafios a serem enfrentados, pois podem afetar diretamente a qualidade atingida pelos modelos utilizados. (SERMANET E CHINTALA, 2012)

Dentro desse contexto, identifica-se os principais riscos os de: **baixa performance**, para modelos de menor complexidade, considerando o grande número de features necessárias para representar as imagens e seus três canais de cores; **confusão** causada por ruídos do dataset, visto que a tarefa de reconhecimento é altamente sensível para ambiguidade visual entre números parecidos; e **baixa verossimilhança entre o dataset e o mundo real**, o que reduziria a usabilidade do modelo

Como forma de mitigar o impacto que os riscos citados possam ter em nossa implementação, serão empregadas técnicas como: *data augmentation,* pre-processamento, e *feature engineering.* Todos estes serão formas de ajudar a aumentar a capacidade de generalização do modelo e, por consequência, aumentar significativamente nossa taxa de acertos.

## 2.4 - Terminologias

- SVHN (Street View House Numbers): base de dados composta por imagens coletadas do Google Street View contendo números de casas reais
- SVHN_medium: subconjunto da base de dados SVHN, com uma amostragem reduzida que visa facilitar a execução em ambientes com menor capacidade computacional
- CNN (Convolutional Neural Network): tipo de rede neural projetada para processar dados visuais, amplamente utilizada em tarefas de reconhecimento de imagens
- Classificação multiclasse: tipo de tarefa de aprendizado supervisionado na qual o modelo é responsável por prever uma entre várias classes possíveis
- F1-score: métrica utilizada para avaliar o desempenho do modelo em contextos com classes desbalanceadas
- ROC-AUC: sigla de Receiver Operating Characteristic - Area Under the Curve, consiste em uma métrica utilizada para avaliar a performance para diferentes limiares de decisão
- Data augmentation: técnica utilizada para aumentar a variedade da base de dados de imagens, ajudando a melhorar a capacidade de generalização do modelo
- Hiperparâmetros: consistem em parâmetros definidos antes do treinamento do modelo que influenciam diretamente no seu desempenho e na sua eficiência

## 2.5 - Custos e Benefícios

Os custo envolvidos são:

- Custo computacional: as imagens presentes no dataset que utilizaremos são coloridas e possuem alta resolução, e o treinamento de modelos com esse tipo específico de imagens exige maior capacidade de processamento, principalmente ao utilizar arquiteturas mais complexas, como redes neurais convolucionais (CNNs)
- Ajuste de hiperparâmetros e tempo de teste: a experimentação com diferentes configurações de modelos demanda múltiplas execuções, o que consome tempo tanto para processamento quanto para análise dos resultados

Os benefícios esperados são:

- Aplicação prática e realista: a identificação automática de números de casas pode ser aplicada em sistemas de entrega, aplicativos de navegação e serviços de emergência. Isso resulta em rotas mais precisas, na redução de falhas humanas e em um melhor atendimento ao público
- Desenvolvimento de solução replicável: a arquitetura do modelo pode ser reutilizada em outras tarefas de classificação de imagens, como reconhecimento de placas de veículos, identificação de fachadas e automação urbana

---

# 3 - Objetivos de *Data Science*

Após a definição dos objetivos de negócio e a avaliação da situação, esta etapa busca estabelecer, do ponto de vista técnico, os propósitos da ciência de dados no projeto. A tradução dos objetivos de negócio para metas operacionais e mensuráveis da modelagem preditiva é essencial para garantir que os resultados obtidos sejam úteis, confiáveis e alinhados com os desafios práticos identificados na Fase 1.

Neste contexto, esta etapa define as metas técnicas, as abordagens metodológicas e os critérios de avaliação que nortearão o desenvolvimento dos modelos. O alinhamento entre os objetivos de negócio e os objetivos de *data science* é fundamental para assegurar que a solução entregue gere valor concreto em aplicações como mapeamento urbano, navegação e resposta a emergências.

## 3.1 - Objetivos de *Data Science*

A partir dos objetivos de negócio previamente definidos (Seção 1.2), o principal objetivo técnico deste projeto é desenvolver um modelo supervisionado de classificação multi-classe capaz de reconhecer com alta precisão os dígitos presentes nas imagens do dataset SVHN_medium, contribuindo para a automação da leitura de endereços em contextos urbanos.

Mais especificamente, espera-se que o modelo aprenda padrões visuais a partir dos 3072 atributos numéricos correspondentes aos pixels RGB das imagens; Generalize bem para condições adversas comuns em imagens urbanas (variações de iluminação, ângulos, ruído e distorções); Seja testado com diferentes arquiteturas; Seja comparado a modelos base, permitindo avaliar os ganhos de desempenho e custo computacional das diferentes abordagens; Explore estratégias como *data augmentation*, regularização e, quando aplicável, *transfer learning*, com foco em aumentar a robustez do modelo; Produza visualizações interpretáveis, como matrizes de confusão, gráficos de desempenho por classe e curvas ROC-AUC, para facilitar o diagnóstico dos erros e a comunicação com stakeholders não técnicos; Sirva como base para experimentações futuras com *pipelines* de pré-processamento, engenharia de atributos e técnicas de busca e ajuste de hiperparâmetros.

## **3.2 - Critério de sucesso do Data Science**

A etapa de ciência de dados será considerada bem-sucedida se os modelos treinados atenderem aos seguintes critérios técnicos e de negócio:

- F1-score macro ≥ 0,80 no conjunto de teste, garantindo desempenho médio elevado entre todas as classes, com especial atenção às menos representadas (criteriosamente alinhado ao KPI de cobertura da Seção 1.3);
- Recall ≥ 0,80 para classes minoritárias, contribuindo para reduzir erros de omissão críticos em aplicações como rotas de emergência ou entregas;
- Boas avaliações em demais métricas utilizadas, como acurácia, precisão e F1-Score;
- Tempo de inferência baixo por imagem em CPU padrão, visando viabilidade de uso em sistemas urbanos embarcados ou tempo real;
- Capacidade de generalização, validada por meio de validação cruzada estratificada (k=5) e simulações com imagens aumentadas artificialmente;
- Robustez frente a perturbações, avaliada por meio da análise detalhada de curvas ROC-AUC, distribuição dos erros nas matrizes de confusão e métricas por classe;
- Entrega de um modelo funcional, documentado e reprodutível, com código versionado no GitHub, scripts de treinamento e avaliação, e visualizações interativas que facilitem a compreensão dos resultados.

---

# 4 - Plano de Projeto

O plano a seguir detalha as etapas do CRISP-DM para esse projeto.

1. **Compreensão do Negócio**
    - Definição do escopo e dos objetivos de negócio.
    - Identificação de potenciais casos de uso.
    - Levantamento das métricas de sucesso em termos qualitativos.
2. **Compreensão dos dados**
    - Download e inspeção inicial do dataset.
    - Análise exploratória para mapear tipos de variáveis, distribuição de classes e eventuais problemas.
    - Documentação das principais características e possíveis ajustes.
3. **Preparação dos Dados**
    - Planejamento das etapas de limpeza e transformação (normalização, tratamento de valores faltantes, etc.).
    - Definição de estratégias de divisão em treino, validação e teste.
    - Escolha preliminar de técnicas de enriquecimento.
4. **Modelagem**
    - Seleção de modelos para avaliar .
    - Estabelecimento de processo de busca por hiperparâmetros de forma iterativa.
    - Registro de configurações e resultados de forma organizada.
5. Avaliação
    - Definição de métricas de desempenho (por ex., recall, F1-score, tempo de inferência).
    - Comparação de resultados entre modelos e identificação de trade-offs.
    - Visualização geral das performances para apoiar a tomada de decisão.
6. **Implantação**
    - Planejamento de um protótipo mínimo (notebook demonstrativo ou API simples).
    - Documentação básica de uso e orientações para testes iniciais.
    - Considerações sobre monitoramento e manutenção.

### 4.1 - **Avaliação Inicial, Ferramentas e Técnicas**

O projeto será implementado em Python, com o desenvolvimento e experimentação realizados em Google Colab ou Jupyter, e todo o código versionado e gerenciado via repositório no GitHub.

---

# 5 - Referências

Ao realizarmos uma busca pelo uso do conjunto de dados, o que encontramos massivamente foi a utilização de sua versão completa, dessa forma, trouxemos referências de uso do uso do conjunto completo.

## 5.1 - **InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets**

Este artigo apresentou o InfoGAN, uma extensão informativa do Generative Adversarial Network (GAN) que aprende representações interpretáveis maximizando a informação mútua entre um subconjunto das variáveis latentes e a observação. Algoritmos como o InfoGAN, são caracterizados para identificação de estilos, features, dessa forma, o SVHN, é uma boa alternativa, pois a tipografia que presente nas casas é diversa, para um mesmo caractere. Como resultado, o InfoGAN conseguiu separar estilos de escrita de formas de dígitos, demonstrando a capacidade do modelo em aprender representações interpretáveis.

## 5.2 - Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks

No artigo "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks", os autores introduzem a arquitetura DCGAN (Deep Convolutional Generative Adversarial Networks), que trouxe maior estabilidade ao treinamento de GANs e permitiu o aprendizado de hierarquias de características visuais de forma não supervisionada.

O dataset SVHN foi um dos principais campos de prova para a qualidade das representações aprendidas pelo modelo. Após treinar a DCGAN no conjunto de treinamento do SVHN sem usar os rótulos (de forma não supervisionada), os autores utilizaram as camadas convolucionais do discriminador da rede como um extrator de características. As imagens do conjunto de teste do SVHN foram passadas por este extrator, e um classificador linear simples (L2-SVM) foi treinado sobre essas características extraídas. O resultado alcançou uma precisão de classificação de 82,8%, o que representou um avanço significativo para métodos não supervisionados na época. Este experimento foi crucial para demonstrar que a DCGAN não apenas gerava imagens realistas, mas também aprendia características semânticas úteis para tarefas de classificação, validando a qualidade da aprendizagem de representação.

## 5.3 - Convolutional Neural Networks Applied to House Numbers Digit Classification

Os autores aplicam redes neurais convolucionais (ConvNets) ao problema de classificação de dígitos extraídos de imagens de fachadas de casas (SVHN), propondo inovações como o pooling Lᵖ (variando p de 1 a ∞) e a incorporação de características de múltiplos estágios (multi-stage features) para enriquecer a representação aprendida. Com uma arquitetura de duas camadas de convolução seguidas de normalização e pooling, alimentando um classificador não-linear de duas camadas, eles usam amostras de treino, extra e validação estratificada e pré-processamento de contraste local/global. Nos testes finais, alcançam 94,85 % de acurácia — um avanço de 4,25 pontos percentuais sobre o melhor anterior (90,6 %) — demonstrando a superioridade do pooling L⁴ e impactos marginais do multi-stage em comparação a tarefas mais texturizadas. Além disso, exploram métricas de validação em função de p no pooling e discutem potenciais extensões para aprendizado não supervisionado e transformações de escala nos dados .

# 6 - Referências Bibliográficas

Chen, X., Duan, Y., Houthooft, R., Schulman, J., Sutskever, I., & Abbeel, P. (2016). *InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets*. arXiv. [https://arxiv.org/abs/1606.03657](https://arxiv.org/abs/1606.03657)

Radford, A., Metz, L., & Chintala, S. (2016). Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. In *Proceedings of the 4th International Conference on Learning Representations (ICLR)*. [https://arxiv.org/abs/1511.06434](https://arxiv.org/abs/1511.06434)

Sermanet, P., Chintala, S., & LeCun, Y. (2012, November). Convolutional neural networks applied to house numbers digit classification. In *Proceedings of the 21st international conference on pattern recognition (ICLR).* [https://arxiv.org/abs/1204.3968](https://arxiv.org/abs/1204.3968)