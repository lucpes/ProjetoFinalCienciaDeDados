# Projeto de Ciência de Dados: Previsão de Preço de Carros Usados

---

## 1. Definição do Problema

### Contexto

O mercado de veículos usados é um dos setores mais dinâmicos da indústria automotiva. A precificação justa de um veículo usado é uma tarefa complexa, influenciada por dezenas de variáveis simultâneas: idade, quilometragem, marca, motor, histórico de proprietários, acidentes, revisões e demanda regional.

Precificar incorretamente um veículo gera perdas para vendedores (preço abaixo do mercado) ou afasta compradores (preço acima). Modelos preditivos baseados em dados históricos têm o potencial de tornar esse processo mais preciso, rápido e imparcial.

### Objetivo

Construir um modelo de machine learning capaz de **prever o preço de venda de um veículo usado** com base em suas características técnicas, históricas e contextuais.

### Tipo de problema

- **Aprendizado supervisionado de regressão**
- **Variável alvo (target):** `Price` — preço estimado de venda em moeda local

### Pergunta norteadora

> *"Dado um conjunto de atributos de um veículo usado, qual é o seu preço justo de mercado?"*

### Critério de sucesso

O modelo será avaliado pela métrica **RMSE (Root Mean Squared Error)** — quanto menor, melhor. Métricas complementares: R² (coeficiente de determinação) e MAE (Erro Médio Absoluto).

---

## 2. Arquitetura Lógica

### 2.1 Visão geral da arquitetura

```
[Fonte de Dados]
       │
       ▼
[Ingestão & Armazenamento]
       │
       ▼
[Processamento & ETL]
       │
       ▼
[Feature Engineering]
       │
       ▼
[Modelagem ML]  ← Parte do Lucas
       │
       ▼
[Visualização]  ← Parte do Lucas
```

### 2.2 Fontes de dados

| Fonte | Descrição | Formato | Tamanho |
|---|---|---|---|
| `used_car_price_prediction_1M.csv` | Dataset principal com mais de 1 milhão de registros de carros usados | CSV | ~1.005.000 linhas × 20 colunas |

**Origem:** Kaggle — Used Car Price Prediction Dataset (simulação de dados de produção com imperfeições intencionais).

### 2.3 Componentes do pipeline

| Componente | Responsabilidade | Tecnologia sugerida |
|---|---|---|
| **Ingestão** | Leitura e validação inicial do CSV | Python + Pandas |
| **Armazenamento intermediário** | Dados brutos e processados | Arquivos Parquet (local) ou SQLite |
| **ETL/ELT** | Limpeza, transformação e engenharia de features | Python + Pandas + NumPy |
| **Modelagem** | Treinamento e avaliação do modelo | Scikit-learn / XGBoost |
| **Visualização** | Análise exploratória e resultado do modelo | Matplotlib / Seaborn / Plotly |

### 2.4 Recursos computacionais necessários

| Recurso | Mínimo recomendado |
|---|---|
| RAM | 8 GB (dataset tem ~750 MB em memória) |
| CPU | Quad-core (treinamento pode levar alguns minutos) |
| Armazenamento | 2 GB livres para dados brutos + processados |
| Ambiente | Python 3.9+ com Jupyter Notebook ou VS Code |

### 2.5 Bibliotecas Python necessárias

```
pandas >= 1.5
numpy >= 1.23
scikit-learn >= 1.2
matplotlib >= 3.6
seaborn >= 0.12
plotly >= 5.0       (opcional, para visualizações interativas)
xgboost >= 1.7      (a critério do Lucas para modelagem)
pyarrow >= 11.0     (para salvar em Parquet)
```

---

## 3. Modelagem de Dados (ETL/ELT)

### 3.1 Diagnóstico da base de dados (resultado da análise executada)

**Dimensões originais:** 1.005.000 linhas × 20 colunas
**Após limpeza:** 1.000.000 linhas × 29 features numéricas (prontas para ML)

**Variáveis e tipos:**

| Feature | Tipo | Valores faltantes | Observações |
|---|---|---|---|
| Brand | Categórica | 0 | 13 marcas únicas |
| Model | Categórica | 0 | 39 modelos únicos |
| Year | Numérica int | 0 | 2000–2024 → removida (substituída por Vehicle_Age) |
| Mileage_kmpl | Numérica float | 80.390 (8%) | Mediana imputada: 18,00 |
| Engine_CC | Numérica float | 80.403 (8%) | Mediana imputada: 1.500,20 |
| Horsepower | Numérica float | 80.403 (8%) | Mediana imputada: 120,06 |
| Fuel_Type | Categórica | 78.733 (7,8%) | Typos corrigidos: 'hybridd'→'Hybrid', 'electrik'→'Electric' |
| Transmission | Categórica | 80.385 (8%) | Manual (59,8%) / Automatic (32,2%) / Unknown (8%) |
| Owner_Type | Categórica | 0 | First (59,9%) / Second (25%) / Third (10%) / Fourth+ (5%) |
| Color | Categórica | 80.419 (8%) | 8 cores únicas |
| City | Categórica | 80.430 (8%) | 9 cidades únicas |
| Kms_Driven | Numérica int | 0 | Quilometragem total |
| Insurance_Valid | Binária int | 0 | 0 ou 1 |
| Service_History | Binária int | 0 | 0 ou 1 |
| Accidents | Numérica int | 0 | Número de acidentes |
| Tax_Paid | Binária int | 0 | 0 ou 1 |
| Number_of_Doors | Numérica int | 0 | 4 ou 5 portas |
| Seats | Numérica int | 0 | Número de assentos |
| Registration_Age | Numérica int | 0 | Removida (redundante com Vehicle_Age) |
| **Price** | **Numérica float (target)** | **0** | Mín: 50.000 / Máx: 37,8M / Mediana: 759K → transformado em Price_log |

**Observações identificadas na execução:**
- **5.000 duplicatas** encontradas e removidas (exatamente 0,5% — padrão intencional do dataset).
- Os **8% de valores faltantes** ocorrem nas mesmas 7 colunas simultaneamente, sugerindo registros com cadastro incompleto.
- Erros tipográficos reais encontrados em `Fuel_Type`: `'hybridd'`, `'electrik'`, `'PETROL'`, `' Diesel'` (com espaço) — totalizando ~6.653 registros corrigidos.
- `Price` tem distribuição extremamente assimétrica (média 1M, máx 37,8M). Apenas **0,58% são outliers extremos** (acima de 4,1M). Optamos por manter todos os registros e aplicar transformação logarítmica.
- `Registration_Age` e `Year` são redundantes — ambas removidas e substituídas pela feature criada `Vehicle_Age`.

---

### 3.2 Plano de ETL

O pipeline ETL será estruturado nas etapas abaixo.

---

#### Etapa 1 — Ingestão e validação inicial

```python
# =============================================================================
# ETAPA 1 — INGESTÃO E VALIDAÇÃO INICIAL
# Carregamos o arquivo CSV e exibimos informações básicas para entender os dados
# =============================================================================
print("=" * 60)
print("ETAPA 1 — Ingestão e validação inicial")
print("=" * 60)

df = pd.read_csv("used_car_price_prediction_1M.csv")

print(f"Linhas: {len(df):,}")
print(f"Colunas: {df.shape[1]}")
print("\nTipos de dados:")
print(df.dtypes)
print("\nValores faltantes:")
missing = df.isnull().sum()
print(missing[missing > 0])
print("\nEstatísticas do target (Price):")
print(df["Price"].describe().apply(lambda x: f"{x:,.2f}"))
```

**Saída esperada:** DataFrame com 1.005.000 linhas × 20 colunas carregado em memória.

---

#### Etapa 2 — Remoção de duplicatas

```python
# =============================================================================
# ETAPA 2 — REMOÇÃO DE DUPLICATAS
# Registros idênticos atrapalham o modelo (ele "aprende" o mesmo caso duas vezes)
# =============================================================================
print("\n" + "=" * 60)
print("ETAPA 2 — Remoção de duplicatas")
print("=" * 60)

antes = len(df)
df = df.drop_duplicates()
depois = len(df)
print(f"Duplicatas removidas: {antes - depois:,}")
print(f"Registros restantes:  {depois:,}")
```

---

#### Etapa 3 — Tratamento de valores faltantes

Estratégia por tipo de variável:

| Tipo de variável | Estratégia |
|---|---|
| Numéricas contínuas (Mileage, Engine_CC, Horsepower) | Imputação pela **mediana** (resistente a outliers) |
| Categóricas (Fuel_Type, Transmission, Color, City) | Imputação pela **moda** ou criação de categoria `"Unknown"` |

```python
# =============================================================================
# ETAPA 3 — TRATAMENTO DE VALORES FALTANTES
# + ou - 8% dos registros têm valores ausentes em 7 colunas.
# Estratégia: mediana para números (resistente a outliers),
# 'Unknown' para textos (preserva a informação de que estava vazio)
# =============================================================================
print("\n" + "=" * 60)
print("ETAPA 3 — Tratamento de valores faltantes")
print("=" * 60)

# Numéricas: preencher com a mediana
numeric_cols = ["Mileage_kmpl", "Engine_CC", "Horsepower"]
for col in numeric_cols:
    mediana = df[col].median()
    df[col] = df[col].fillna(mediana)
    print(f"{col}: mediana = {mediana:.2f}")

# Categóricas: preencher com 'Unknown'
categorical_missing = ["Fuel_Type", "Transmission", "Color", "City"]
for col in categorical_missing:
    df[col] = df[col].fillna("Unknown")
    print(f"{col}: preenchido com 'Unknown'")

print(f"\nValores faltantes restantes: {df.isnull().sum().sum()}")
```

---

#### Etapa 4 — Tratamento de outliers

A variável `Price` apresenta valores extremos (máx ~37,8M com mediana ~759K). Abordar com:

1. **Análise via IQR (Interquartile Range):** identificar e remover ou winsorizar registros acima do limite superior.
2. **Transformação logarítmica do target:** aplicar `log(Price)` para normalizar a distribuição — uma técnica padrão em problemas de previsão de preços.

```python
# =============================================================================
# ETAPA 4 — TRATAMENTO DE OUTLIERS E TRANSFORMAÇÃO DO TARGET
# O preço vai de 50 mil a 37,8 milhões — distribuição extremamente assimétrica.
# Solução: transformação logarítmica (log(Price)) que comprime a escala.
# O modelo aprende na escala log; para converter de volta: np.expm1(predição)
# =============================================================================
print("\n" + "=" * 60)
print("ETAPA 4 — Transformação do target (Price)")
print("=" * 60)

Q1  = df["Price"].quantile(0.25)
Q3  = df["Price"].quantile(0.75)
IQR = Q3 - Q1
limite_superior = Q3 + 3 * IQR
outliers = (df["Price"] > limite_superior).sum()
print(f"Outliers extremos detectados (acima de {limite_superior:,.0f}): {outliers:,} ({outliers/len(df)*100:.2f}%)")
print("→ Optamos por manter e usar transformação logarítmica")

df["Price_log"] = np.log1p(df["Price"])
print(f"\nPrice_log — média: {df['Price_log'].mean():.4f}, std: {df['Price_log'].std():.4f}")
print("(distribuição muito mais simétrica agora)")
```

> **Nota:** se optar por transformação log no target, lembre-se de reverter a predição com `np.expm1(y_pred)` ao final.

---

#### Etapa 5 — Padronização e correção de variáveis categóricas

O dataset contém erros tipográficos e formatação inconsistente (problema intencional do dataset).

```python
# =============================================================================
# ETAPA 5 — PADRONIZAÇÃO DAS VARIÁVEIS CATEGÓRICAS
# O dataset tem erros intencionais: espaços extras, capitalização errada, typos.
# Exemplos encontrados: 'PETROL', ' Diesel', 'hybridd', 'electrik'
# =============================================================================
print("\n" + "=" * 60)
print("ETAPA 5 — Padronização de variáveis categóricas")
print("=" * 60)

# Padronizar todas as colunas de texto
text_cols = ["Brand", "Model", "Fuel_Type", "Transmission", "Owner_Type", "Color", "City"]
for col in text_cols:
    df[col] = df[col].str.strip().str.title()

# Corrigir erros tipográficos identificados na análise
correcoes = {
    "Fuel_Type": {
        "Hybridd":  "Hybrid",
        "Electrik": "Electric",
        "Cng":      "CNG",
    }
}
for col, mapa in correcoes.items():
    df[col] = df[col].replace(mapa)
    print(f"{col}: {len(mapa)} correções aplicadas → {sorted(df[col].unique())}")
```

---

#### Etapa 6 — Engenharia de features (Feature Engineering)

Criar novas variáveis que potencialmente aumentam o poder preditivo do modelo:

```python
# =============================================================================
# ETAPA 6 — FEATURE ENGINEERING (Engenharia de Features)
# Criamos novas variáveis que podem ajudar o modelo a aprender melhor.
# Cada nova feature captura uma relação que não estava explícita nos dados.
# =============================================================================
print("\n" + "=" * 60)
print("ETAPA 6 — Feature Engineering")
print("=" * 60)

# Idade real do veículo (mais intuitivo que Registration_Age)
df["Vehicle_Age"] = 2025 - df["Year"]
print("✓ Vehicle_Age: idade do veículo em anos")

# Indicador de marca premium (veículos de luxo têm precificação diferente)
marcas_premium = ["Bmw", "Mercedes", "Audi", "Porsche", "Jaguar", "Land Rover", "Volvo", "Lexus"]
df["Is_Premium"] = df["Brand"].isin(marcas_premium).astype(int)
print(f"✓ Is_Premium: {df['Is_Premium'].sum():,} veículos premium ({df['Is_Premium'].mean()*100:.1f}%)")

# Potência por cilindrada: mede eficiência mecânica do motor
df["Power_per_CC"] = df["Horsepower"] / (df["Engine_CC"] + 1)
print(f"✓ Power_per_CC: média = {df['Power_per_CC'].mean():.4f}")

# Quilometragem por ano: uso anual médio (quilômetro/ano)
df["Kms_per_Year"] = df["Kms_Driven"] / (df["Vehicle_Age"] + 1)
print(f"✓ Kms_per_Year: média = {df['Kms_per_Year'].mean():,.0f} km/ano")

# Score de confiabilidade: combina seguros, histórico e acidentes
df["Reliability_Score"] = (
    df["Insurance_Valid"] +
    df["Service_History"] +
    df["Tax_Paid"]        -
    df["Accidents"]
)
print(f"✓ Reliability_Score: varia de {df['Reliability_Score'].min()} a {df['Reliability_Score'].max()}")
```

---

#### Etapa 7 — Codificação de variáveis categóricas

Para que o modelo de ML possa processar variáveis categóricas:

```python
# =============================================================================
# ETAPA 7 — ENCODING DAS VARIÁVEIS CATEGÓRICAS
# Algoritmos de ML só entendem números. Precisamos converter texto em números.
#
# Duas estratégias:
# - One-Hot Encoding (OHE): para colunas com poucas categorias (ex: Fuel_Type)
#   → cria uma coluna binária (0/1) para cada categoria
# - Label Encoding: para colunas com muitas categorias (ex: Brand, Model)
#   → substitui cada texto por um número inteiro
# =============================================================================
print("\n" + "=" * 60)
print("ETAPA 7 — Encoding de variáveis categóricas")
print("=" * 60)

# One-Hot Encoding: baixa cardinalidade
ohe_cols = ["Fuel_Type", "Transmission", "Owner_Type"]
df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)
print(f"✓ OHE aplicado em: {ohe_cols}")

# Label Encoding: alta cardinalidade
label_encoders = {}
le_cols = ["Brand", "Model", "City", "Color"]
for col in le_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    df = df.drop(columns=[col])
print(f"✓ Label Encoding aplicado em: {le_cols}")

# Remover colunas redundantes ou desnecessárias
# Year → substituído por Vehicle_Age
# Registration_Age → redundante com Vehicle_Age
# Price → usaremos Price_log como target
df = df.drop(columns=["Year", "Registration_Age", "Price"])
print(f"\nShape final: {df.shape}")
print(f"Colunas: {list(df.columns)}")
```

---

#### Etapa 8 — Divisão treino/teste e normalização

```python
# =============================================================================
# ETAPA 8 — DIVISÃO TREINO / TESTE
# Dividimos os dados em 80% para treino e 20% para teste.
# O modelo aprende com o treino e é avaliado no teste (dados que nunca viu).
# random_state=42 garante que a divisão seja sempre a mesma (reprodutibilidade)
# =============================================================================
print("\n" + "=" * 60)
print("ETAPA 8 — Divisão treino / teste")
print("=" * 60)

X = df.drop(columns=["Price_log"])   # features (entradas)
y = df["Price_log"]                  # target (saída)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Treino: {len(X_train):,} registros (80%)")
print(f"Teste:  {len(X_test):,} registros (20%)")
print(f"Features: {X.shape[1]}")
print(f"\nMédia do target — Treino: {y_train.mean():.4f} | Teste: {y_test.mean():.4f}")
print("(médias similares = divisão bem distribuída ✓)")
```

---

#### Etapa 9 — Exportação dos dados processados

```python
# =============================================================================
# ETAPA 9 — EXPORTAR DADOS PROCESSADOS
# Salvamos tudo para que o colega possa continuar com ML e Visualização.
# =============================================================================
print("\n" + "=" * 60)
print("ETAPA 9 — Exportando arquivos")
print("=" * 60)

# Dataset completo processado (formato Parquet = mais eficiente que CSV)
df.to_parquet("data_processed.parquet", index=False)
print(f"✓ data_processed.parquet ({os.path.getsize('data_processed.parquet')/1024**2:.1f} MB)")

# Splits prontos para o modelo
with open("train_test_split.pkl", "wb") as f:
    pickle.dump((X_train, X_test, y_train, y_test), f)
print(f"✓ train_test_split.pkl ({os.path.getsize('train_test_split.pkl')/1024**2:.1f} MB)")
```

---

### 3.3 Resumo do fluxo ETL (executado)

```
CSV bruto (1.005.000 × 20)
        │
        ├── Etapa 1: Ingestão e validação
        ├── Etapa 2: Remoção de 5.000 duplicatas → 1.000.000 registros
        ├── Etapa 3: Imputação de faltantes (mediana em numéricas, 'Unknown' em categóricas)
        ├── Etapa 4: Log transform em Price → criação de Price_log
        ├── Etapa 5: Padronização + correção de typos (6.653 registros corrigidos)
        ├── Etapa 6: Feature Engineering → 5 novas variáveis
        ├── Etapa 7: Encoding (OHE + Label Encoding) → 29 features numéricas
        ├── Etapa 8: Split treino/teste (80/20)
        └── Etapa 9: Exportação dos 3 arquivos de saída
                │
                ├── used_car_price_prediction_1M.csv  → dado bruto original
                ├── data_processed.parquet            → dataset limpo (53 MB)
                └── train_test_split.pkl              → splits prontos (185 MB)
                                │
                                ▼
                        Modelo ML (Lucas) → Visualização (Lucas)
```

### 3.4 Arquivos de saída para o pipeline de ML

O ETL produziu 3 arquivos que formam a interface entre esta etapa e o trabalho do Lucas:

| Arquivo | Formato | Tamanho | Conteúdo |
|---|---|---|---|
| `used_car_price_prediction_1M.csv` | CSV | ~750 MB em memória | Dado bruto original — referência |
| `data_processed.parquet` | Parquet | 53 MB | Dataset completo após limpeza, com 1.000.000 linhas e 30 colunas (29 features + target `Price_log`). Parquet é armazenamento colunar comprimido — mais rápido e ~14× menor que o CSV equivalente. |
| `train_test_split.pkl` | Pickle | 185 MB | Objeto Python com os 4 splits já separados: `X_train` (800k), `X_test` (200k), `y_train`, `y_test`. Garante reprodutibilidade — o colega usa exatamente a mesma divisão. |

**Lucas, como carrega os arquivos:**

```python
import pandas as pd
import pickle
import numpy as np

# Opção A: carregar dataset completo (mais flexível)
df = pd.read_parquet("data_processed.parquet")
X = df.drop(columns=["Price_log"])
y = df["Price_log"]

# Opção B: carregar splits prontos (mais rápido para começar)
with open("train_test_split.pkl", "rb") as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

# IMPORTANTE: converter predições de volta para escala original
# y_pred está em log → aplicar expm1 para obter o preço real
preco_previsto = np.expm1(modelo.predict(X_test))
```

---

## 4. Entregáveis desta fase (itens 1–3)

- [x] Definição clara do problema e objetivo
- [x] Arquitetura lógica do pipeline
- [x] Script ETL documentado e comentado (`etl_carros_usados.py`)
- [x] Diagnóstico completo da base de dados com números reais
- [x] ETL executado: 1.005.000 → 1.000.000 registros limpos, 29 features
- [x] Dataset processado exportado (`data_processed.parquet` — 53 MB)
- [x] Splits de treino e teste exportados (`train_test_split.pkl` — 185 MB)

## 5. O que você precisa para continuar (itens 4–5)

Pegue os seguintes arquivos e prossiga:

1. **`data_processed.parquet`** — dataset limpo e com features prontas
2. **`train_test_split.pkl`** — splits já separados (para garantir reprodutibilidade)
3. **Target:** `Price_log` (log do preço) — lembrar de aplicar `np.expm1()` nas predições para converter de volta
4. **Métrica de avaliação:** RMSE (principal) + R² + MAE
5. **Sugestão de modelos:** Random Forest, Gradient Boosting (XGBoost/LightGBM), ou Regressão Linear como baseline

---
