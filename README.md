# Classification Multilabel - Projet ML

Projet de classification multilabel pour l'analyse de problèmes de programmation. Ce projet permet d'entraîner et d'évaluer différents modèles de machine learning pour classifier automatiquement des problèmes selon plusieurs catégories.

## 📁 Structure du Projet

```
ml-classification-project/
│
├── cli.py                      # Interface en ligne de commande (CLI)
├── requirements.txt            # Dépendances Python
├── README.md                   # Documentation
│
├── src/                        # Code source
│   ├── __init__.py            # Initialisation du module
│   ├── data.py                # Preprocessing des données
│   ├── models.py              # Classes de modèles
│   ├── evaluation.py          # Calcul des métriques
│   └── train_model.py         # Entraînement des modèles
│
├── data/                       # Données brutes (fichiers JSON)
│
├── saved_models/              # Modèles entraînés sauvegardés
│   ├── tfidf_logreg/
│   └── tfidf_minilm/
│
├── reports/                    # Rapports d'évaluation
│   ├── tfidf_logreg/
│   │   ├── confusion_matrices/
│   │   ├── report_TfidfLogReg.pdf
│   │   └── metrics_per_class.csv
│   └── tfidf_minilm/
│
└── confusion_matrices/        # Matrices de confusion
```

## 🚀 Installation

### 1. Cloner le projet
```bash
git clone <votre-repo>
cd ml-classification-project
```

### 2. Créer un environnement virtuel (recommandé)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

## 📊 Utilisation de la CLI

### Commandes Principales

#### 1. Entraîner un modèle TF-IDF + Logistic Regression
```bash
python cli.py train --model tfidf-logreg --data-dir ./data
```

#### 2. Entraîner un modèle TF-IDF + MiniLM
```bash
python cli.py train --model tfidf-minilm --data-dir ./data
```

#### 3. Entraîner avec augmentation des classes rares
```bash
python cli.py train --model tfidf-minilm --data-dir ./data --augment --custom-thresholds
```

#### 4. Prédire sur de nouvelles questions
```bash
# Prédire une ou plusieurs questions directement
python cli.py predict --model-dir ./saved_models/tfidf_logreg \
  --questions "What is the derivative of x^2?" "How to solve linear equations?"

# Prédire depuis un fichier JSON
python cli.py predict --model-dir ./saved_models/tfidf_minilm \
  --input-file questions.json --output-file predictions.json

# Avec mode verbose pour voir les probabilités
python cli.py predict --model-dir ./saved_models/tfidf_minilm \
  --questions "What is conditional probability?" --verbose

# Limiter au top 3 tags les plus probables
python cli.py predict --model-dir ./saved_models/tfidf_logreg \
  --input-file questions.json --top-k 3 --verbose

# Avec seuil de probabilité personnalisé
python cli.py predict --model-dir ./saved_models/tfidf_minilm \
  --questions "What is Bayes theorem?" --threshold 0.3 --verbose
```

**Formats supportés pour les fichiers d'entrée:**

```json
// Format 1: Liste simple de questions
[
  "What is the probability of rolling a 6?",
  "How to solve linear equations?",
  "Explain graph traversal algorithms"
]

// Format 2: Liste de dictionnaires
[
  {"question": "What is the probability of rolling a 6?", "id": 1},
  {"question": "How to solve linear equations?", "id": 2}
]
```

**Format de sortie (JSON):**
```json
[
  {
    "question": "What is the probability of rolling a 6?",
    "predicted_tags": ["probabilities", "math"],
    "probabilities": {
      "probabilities": 0.8532,
      "math": 0.6421
    }
  }
]
```

#### 5. Comparer les deux modèles
```bash
python cli.py compare --data-dir ./data
```

### Options Avancées

```bash
# Paramètres de split
python cli.py train --model tfidf-logreg --data-dir ./data --test-size 0.3 --random-state 123

# Paramètres du modèle
python cli.py train --model tfidf-logreg --data-dir ./data --max-features 50000 --max-iter 300

# Personnaliser les répertoires de sortie
python cli.py train --model tfidf-logreg --data-dir ./data --save-dir ./my_models --report-dir ./my_reports

# Spécifier les tags rares à augmenter
python cli.py train --model tfidf-minilm --data-dir ./data --augment --rare-tags probabilities games
```

### Aide
```bash
# Afficher l'aide générale
python cli.py --help

# Afficher l'aide pour une commande
python cli.py train --help
python cli.py compare --help
python cli.py predict --help
```

## 🔧 Modules Python

### 1. `data.py` - Preprocessing des données

**Fonctions principales:**

- `load_json_files(data_dir)`: Charge tous les fichiers JSON
- `preprocess_data(df)`: Pipeline complet de preprocessing
- `clean_text(text)`: Nettoie et normalise le texte
- `augment_rare_classes(X_train, Y_train, rare_tags)`: Augmente les classes rares

**Exemple d'utilisation:**
```python
from src.data import load_json_files, preprocess_data

# Charger les données
df_raw = load_json_files("data/")

# Preprocesser
df = preprocess_data(df_raw)
```

### 2. `models.py` - Classes de modèles

**Classes disponibles:**

#### TfidfLogRegClassifier
Modèle basé sur TF-IDF + Logistic Regression

```python
from src.models import TfidfLogRegClassifier

model = TfidfLogRegClassifier(max_features=30000, max_iter=200)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
model.save("saved_models/my_model")
```

#### TfidfMiniLMClassifier
Modèle basé sur TF-IDF + MiniLM embeddings + Logistic Regression

```python
from src.models import TfidfMiniLMClassifier

model = TfidfMiniLMClassifier(max_features=30000, max_iter=300)
model.fit(X_train, y_train)

# Prédiction avec seuils personnalisés
thresholds = {'probabilities': 0.3, 'games': 0.3, 'geometry': 0.3}
predictions = model.predict_with_threshold(X_test, thresholds)
```

### 3. `evaluation.py` - Évaluation des modèles

**Fonctions principales:**

- `evaluate_model(model, X_test, Y_test, class_names)`: Pipeline complet d'évaluation
- `compute_classification_report(Y_true, Y_pred, class_names)`: Calcule les métriques
- `generate_confusion_matrices(Y_true, Y_pred, class_names)`: Génère les matrices
- `generate_pdf_report(...)`: Crée un rapport PDF

**Exemple d'utilisation:**
```python
from src.evaluation import evaluate_model

results = evaluate_model(
    model=my_model,
    X_test=X_test,
    Y_test=Y_test,
    class_names=['math', 'graphs', 'strings'],
    output_dir="reports/my_evaluation",
    model_name="MyModel"
)

# Accéder aux résultats
print(results['report_text'])
print(results['additional_metrics'])
```

### 4. `train_model.py` - Entraînement et Prédiction

**Fonctions principales:**

- `train_tfidf_logreg(data_dir, ...)`: Entraîne TF-IDF + LogReg
- `train_tfidf_minilm(data_dir, ...)`: Entraîne TF-IDF + MiniLM
- `compare_models(data_dir, ...)`: Compare les modèles
- `predict_questions(questions, model_dir, ...)`: Prédit les tags pour de nouvelles questions

**Exemple d'utilisation - Entraînement:**
```python
from src.train_model import train_tfidf_logreg

results = train_tfidf_logreg(
    data_dir="data/",
    test_size=0.2,
    random_state=42,
    save_dir="saved_models/my_model",
    max_features=30000,
    max_iter=200
)

model = results['model']
evaluation_results = results['results']
```

**Exemple d'utilisation - Prédiction:**
```python
from src.train_model import predict_questions

# Prédire sur une seule question
predictions = predict_questions(
    questions="What is the derivative of x^2?",
    model_dir="saved_models/tfidf_logreg"
)

# Prédire sur plusieurs questions avec options
predictions = predict_questions(
    questions=[
        "What is conditional probability?",
        "How to implement a binary search tree?",
        "Explain Bayes theorem"
    ],
    model_dir="saved_models/tfidf_minilm",
    threshold=0.3,  # Seuil personnalisé
    top_k=5        # Limiter à 5 tags maximum
)

# Accéder aux résultats
for pred in predictions:
    print(f"Tags: {pred['tags']}")
    print(f"Probabilities: {pred['probabilities']}")
```

**Paramètres de `predict_questions`:**
- `questions` (str ou List[str]): Question(s) à classifier
- `model_dir` (str): Répertoire contenant le modèle sauvegardé
- `threshold` (float, optional): Seuil de probabilité (défaut: 0.5 ou seuils du modèle)
- `top_k` (int, optional): Nombre maximum de tags à retourner

**Format de retour:**
```python
[
    {
        'tags': ['probabilities', 'math'],
        'probabilities': {
            'probabilities': 0.8532,
            'math': 0.6421
        }
    },
    ...
]
```

## 📈 Classes Cibles

Le projet supporte la classification selon les catégories suivantes:
- `math`: Problèmes mathématiques
- `graphs`: Théorie des graphes
- `strings`: Manipulation de chaînes
- `number theory`: Théorie des nombres
- `trees`: Structures d'arbres
- `geometry`: Géométrie
- `games`: Théorie des jeux
- `probabilities`: Probabilités

## 📊 Métriques d'Évaluation

Le projet calcule automatiquement:
- **Precision, Recall, F1-Score** par classe et moyennes
- **Hamming Loss**: Proportion d'erreurs sur l'ensemble des labels
- **Exact Match Ratio**: Proportion de prédictions parfaitement correctes
- **Matrices de confusion** pour chaque classe
- **Support**: Nombre d'exemples par classe

Tous les résultats sont sauvegardés dans:
- Un fichier CSV avec les métriques par classe
- Un rapport PDF complet avec graphiques
- Des images PNG des matrices de confusion

## 🎯 Stratégies d'Amélioration

### Augmentation des Classes Rares
Pour améliorer les performances sur les classes sous-représentées:
```bash
python cli.py train --model tfidf-minilm --data-dir ./data --augment --augment-multiplier 3
```

### Seuils Personnalisés
Ajustez les seuils de classification par classe:
```bash
python cli.py train --model tfidf-minilm --data-dir ./data --custom-thresholds
```

## 🔮 Exemple de Questions pour Test

### Mathématiques
```bash
python cli.py predict --model-dir ./saved_models/tfidf_minilm \
  --questions "What is the derivative of x^2?" \
  "How to solve quadratic equations?" \
  "Explain integration by parts" \
  --verbose
```

### Utilisation avec fichier JSON
```bash
# Créer un fichier questions.json
echo '[
  "What is the probability of getting two heads in a row?",
  "How to traverse a binary tree?",
  "Explain the Euclidean algorithm",
  "What is dynamic programming?"
]' > questions.json

# Prédire et sauvegarder les résultats
python cli.py predict \
  --model-dir ./saved_models/tfidf_minilm \
  --input-file questions.json \
  --output-file predictions.json \
  --verbose \
  --top-k 5
```

## 📝 Notes

- Les modèles sauvegardés incluent le type de modèle pour un chargement automatique
- La fonction `predict_questions` détecte automatiquement le type de modèle (TfidfLogReg ou TfidfMiniLM)
- Le mode `--verbose` affiche les probabilités pour chaque tag prédit
- Les seuils personnalisés permettent d'ajuster la sensibilité pour chaque classe