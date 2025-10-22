"""
Module d'entraînement des modèles de classification multilabel.
Gère le pipeline complet: chargement, preprocessing, entraînement et évaluation.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from typing import Optional, Dict, Tuple, List, Union

# Import des modules du projet
from data import (
    load_json_files, 
    preprocess_data, 
    augment_rare_classes
)
from models import TfidfLogRegClassifier, TfidfMiniLMClassifier
from evaluation import evaluate_model


def train_tfidf_logreg(data_dir: str,
                       test_size: float = 0.2,
                       random_state: int = 42,
                       save_dir: str = "saved_models/tfidf_logreg",
                       report_dir: str = "reports/tfidf_logreg",
                       **model_params) -> Dict:
    """
    Entraîne un modèle TF-IDF + Logistic Regression.
    
    Args:
        data_dir (str): Répertoire contenant les fichiers JSON
        test_size (float): Proportion du jeu de test (entre 0 et 1)
        random_state (int): Seed pour la reproductibilité
        save_dir (str): Répertoire de sauvegarde du modèle
        report_dir (str): Répertoire pour les rapports d'évaluation
        **model_params: Paramètres supplémentaires pour le modèle
        
    Returns:
        Dict: Dictionnaire contenant le modèle entraîné et les résultats
    """
    print("\n" + "="*70)
    print("ENTRAÎNEMENT: TF-IDF + LOGISTIC REGRESSION")
    print("="*70)
    
    # 1. Chargement des données
    print("\n📂 Chargement des données...")
    df_raw = load_json_files(data_dir)
    
    # 2. Preprocessing
    df = preprocess_data(df_raw)
    
    # 3. Extraction des features et labels
    X = df["clean_text"].fillna("")
    y = df["raw"]
    
    print(f"📊 Nombre total d'exemples: {len(X)}")
    print(f"📊 Nombre de classes: {len(set([tag for tags in y for tag in tags]))}")
    
    # 4. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"🔀 Split effectué: {len(X_train)} train, {len(X_test)} test")
    
    # 5. Création et entraînement du modèle
    model = TfidfLogRegClassifier(**model_params)
    model.fit(X_train, y_train)
    
    # 6. Transformation des labels pour l'évaluation
    Y_test = model.mlb.transform(y_test)
    
    # 7. Évaluation
    results = evaluate_model(
        model, 
        X_test, 
        Y_test,
        class_names=list(model.mlb.classes_),
        output_dir=report_dir,
        model_name="TfidfLogReg"
    )
    
    # 8. Sauvegarde du modèle
    model.save(save_dir)
    
    print(f"\n✅ Entraînement terminé avec succès!")
    print(f"📁 Modèle sauvegardé: {save_dir}/")
    print(f"📁 Rapports générés: {report_dir}/")
    
    return {
        'model': model,
        'results': results,
        'X_test': X_test,
        'y_test': y_test,
        'Y_test': Y_test
    }


def train_tfidf_minilm(data_dir: str,
                       test_size: float = 0.2,
                       random_state: int = 42,
                       augment_rare: bool = True,
                       rare_tags: Optional[list] = None,
                       augment_multiplier: int = 2,
                       use_custom_thresholds: bool = True,
                       save_dir: str = "saved_models/tfidf_minilm",
                       report_dir: str = "reports/tfidf_minilm",
                       **model_params) -> Dict:
    """
    Entraîne un modèle TF-IDF + MiniLM + Logistic Regression.
    
    Args:
        data_dir (str): Répertoire contenant les fichiers JSON
        test_size (float): Proportion du jeu de test
        random_state (int): Seed pour la reproductibilité
        augment_rare (bool): Si True, augmente les classes rares
        rare_tags (list): Liste des tags rares à augmenter
        augment_multiplier (int): Facteur de multiplication pour l'augmentation
        use_custom_thresholds (bool): Si True, utilise des seuils personnalisés
        save_dir (str): Répertoire de sauvegarde du modèle
        report_dir (str): Répertoire pour les rapports
        **model_params: Paramètres supplémentaires pour le modèle
        
    Returns:
        Dict: Dictionnaire contenant le modèle entraîné et les résultats
    """
    print("\n" + "="*70)
    print("ENTRAÎNEMENT: TF-IDF + MiniLM + LOGISTIC REGRESSION")
    print("="*70)
    
    # 1. Chargement des données
    print("\n📂 Chargement des données...")
    df_raw = load_json_files(data_dir)
    
    # 2. Preprocessing
    df = preprocess_data(df_raw)
    
    # 3. Extraction des features et labels
    X = df["clean_text"].fillna("")
    y = df["raw"]
    
    print(f"📊 Nombre total d'exemples: {len(X)}")
    
    # 4. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"🔀 Split effectué: {len(X_train)} train, {len(X_test)} test")
    
    # 5. Création du modèle (nécessaire pour obtenir mlb)
    model = TfidfMiniLMClassifier(**model_params)
    
    # Transformation préliminaire pour obtenir les classes
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb_temp = MultiLabelBinarizer()
    Y_train = mlb_temp.fit_transform(y_train)
    Y_test = mlb_temp.transform(y_test)
    
    # 6. Augmentation des classes rares (optionnel)
    if augment_rare:
        if rare_tags is None:
            rare_tags = ["probabilities", "games", "geometry"]
        
        print(f"\n📈 Augmentation des classes rares: {rare_tags}")
        Y_train_df = pd.DataFrame(Y_train, columns=mlb_temp.classes_)
        X_train_aug, Y_train_aug = augment_rare_classes(
            X_train, 
            Y_train_df, 
            rare_tags, 
            multiplier=augment_multiplier
        )
        
        # Reconvertir y_train pour l'entraînement
        y_train_aug = []
        for i in range(len(Y_train_aug)):
            tags = [mlb_temp.classes_[j] for j in range(len(mlb_temp.classes_)) 
                   if Y_train_aug[i][j] == 1]
            y_train_aug.append(tags)
        
        X_train = X_train_aug
        y_train = y_train_aug
    
    # 7. Entraînement du modèle
    model.fit(X_train, y_train)
    
    # 8. Prédiction et évaluation
    if use_custom_thresholds and rare_tags:
        print("\n🎯 Utilisation de seuils personnalisés pour les classes rares")
        thresholds = {cls: 0.3 if cls in rare_tags else 0.5 
                     for cls in model.mlb.classes_}
        Y_pred = model.predict_with_threshold(X_test, thresholds)
        
        # Créer un wrapper pour l'évaluation
        class ModelWrapper:
            def __init__(self, model, thresholds):
                self.model = model
                self.thresholds = thresholds
            
            def predict(self, X):
                return self.model.predict_with_threshold(X, self.thresholds)
        
        eval_model = ModelWrapper(model, thresholds)
    else:
        eval_model = model
    
    # Transformation des labels pour l'évaluation
    Y_test_eval = model.mlb.transform(y_test)
    
    # 9. Évaluation
    results = evaluate_model(
        eval_model, 
        X_test, 
        Y_test_eval,
        class_names=list(model.mlb.classes_),
        output_dir=report_dir,
        model_name="TfidfMiniLM"
    )
    
    # 10. Sauvegarde du modèle
    model.save(save_dir)
    
    print(f"\n✅ Entraînement terminé avec succès!")
    print(f"📁 Modèle sauvegardé: {save_dir}/")
    print(f"📁 Rapports générés: {report_dir}/")
    
    return {
        'model': model,
        'results': results,
        'X_test': X_test,
        'y_test': y_test,
        'Y_test': Y_test_eval,
        'thresholds': thresholds if use_custom_thresholds and rare_tags else None
    }


def compare_models(data_dir: str,
                  test_size: float = 0.2,
                  random_state: int = 42) -> pd.DataFrame:
    """
    Compare les performances de plusieurs modèles.
    
    Args:
        data_dir (str): Répertoire contenant les données
        test_size (float): Proportion du jeu de test
        random_state (int): Seed pour la reproductibilité
        
    Returns:
        pd.DataFrame: DataFrame comparant les métriques des modèles
    """
    print("\n" + "="*70)
    print("COMPARAISON DES MODÈLES")
    print("="*70)
    
    # Entraîner le modèle TF-IDF + LogReg
    print("\n🔹 Modèle 1/2: TF-IDF + LogReg")
    results_logreg = train_tfidf_logreg(
        data_dir,
        test_size=test_size,
        random_state=random_state,
        save_dir="saved_models/comparison/tfidf_logreg",
        report_dir="reports/comparison/tfidf_logreg"
    )
    
    # Entraîner le modèle TF-IDF + MiniLM
    print("\n🔹 Modèle 2/2: TF-IDF + MiniLM")
    results_minilm = train_tfidf_minilm(
        data_dir,
        test_size=test_size,
        random_state=random_state,
        save_dir="saved_models/comparison/tfidf_minilm",
        report_dir="reports/comparison/tfidf_minilm"
    )
    
    # Extraire les métriques macro avg
    logreg_metrics = results_logreg['results']['report_dict']['macro avg']
    minilm_metrics = results_minilm['results']['report_dict']['macro avg']
    
    logreg_add = results_logreg['results']['additional_metrics']
    minilm_add = results_minilm['results']['additional_metrics']
    
    # Créer le DataFrame de comparaison
    comparison_df = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1-Score', 'Hamming Loss', 'Exact Match Ratio'],
        'TF-IDF + LogReg': [
            logreg_metrics['precision'],
            logreg_metrics['recall'],
            logreg_metrics['f1-score'],
            logreg_add['hamming_loss'],
            logreg_add['exact_match_ratio']
        ],
        'TF-IDF + MiniLM': [
            minilm_metrics['precision'],
            minilm_metrics['recall'],
            minilm_metrics['f1-score'],
            minilm_add['hamming_loss'],
            minilm_add['exact_match_ratio']
        ]
    })
    
    # Calculer la différence
    comparison_df['Différence'] = (
        comparison_df['TF-IDF + MiniLM'] - comparison_df['TF-IDF + LogReg']
    )
    
    # Sauvegarder la comparaison
    os.makedirs("reports/comparison", exist_ok=True)
    comparison_df.to_csv("reports/comparison/models_comparison.csv", index=False)
    
    print("\n" + "="*70)
    print("RÉSULTATS DE LA COMPARAISON")
    print("="*70)
    print(comparison_df.to_string(index=False))
    print("="*70)
    
    print(f"\n💾 Comparaison sauvegardée: reports/comparison/models_comparison.csv")
    
    return comparison_df


def predict_questions(questions: Union[str, List[str]],
                     model_dir: str,
                     threshold: Optional[float] = None,
                     top_k: Optional[int] = None) -> List[Dict]:
    """
    Prédit les tags pour une ou plusieurs questions en utilisant un modèle sauvegardé.
    
    Args:
        questions (str or List[str]): Question(s) à classifier
        model_dir (str): Répertoire contenant le modèle sauvegardé
        threshold (float, optional): Seuil de probabilité pour les prédictions
                                     Si None, utilise le seuil par défaut du modèle
        top_k (int, optional): Nombre maximum de tags à retourner par question
        
    Returns:
        List[Dict]: Liste de dictionnaires contenant les prédictions pour chaque question
                   Format: [{'tags': [...], 'probabilities': {...}}, ...]
    """
    # Convertir une seule question en liste
    if isinstance(questions, str):
        questions = [questions]
    
    # Vérifier que le répertoire du modèle existe
    if not os.path.exists(model_dir):
        raise ValueError(f"Le répertoire du modèle n'existe pas: {model_dir}")
    
    # Charger le modèle
    print(f"📥 Chargement du modèle depuis: {model_dir}")
    
    # Détecter le type de modèle
    model_type_file = os.path.join(model_dir, "model_type.txt")
    if os.path.exists(model_type_file):
        with open(model_type_file, 'r') as f:
            model_type = f.read().strip()
    else:
        # Essayer de détecter automatiquement
        if os.path.exists(os.path.join(model_dir, "sentence_transformer")):
            model_type = "TfidfMiniLMClassifier"
        else:
            model_type = "TfidfLogRegClassifier"
    
    print(f"🔍 Type de modèle détecté: {model_type}")
    
    # Charger le modèle approprié
    if model_type == "TfidfLogRegClassifier":
        model = TfidfLogRegClassifier()
        model.load(model_dir)
    elif model_type == "TfidfMiniLMClassifier":
        model = TfidfMiniLMClassifier()
        model.load(model_dir)
    else:
        raise ValueError(f"Type de modèle inconnu: {model_type}")
    
    # Effectuer les prédictions
    predictions = []
    
    for question in questions:
        # Obtenir les probabilités
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba([question])[0]
        else:
            # Fallback si predict_proba n'existe pas
            pred = model.predict([question])[0]
            predictions.append({
                'tags': [model.mlb.classes_[i] for i, val in enumerate(pred) if val == 1],
                'probabilities': {}
            })
            continue
        
        # Créer un dictionnaire de probabilités pour tous les tags
        proba_dict = {
            tag: float(prob) 
            for tag, prob in zip(model.mlb.classes_, proba)
        }
        
        # Appliquer le seuil
        if threshold is not None:
            # Utiliser le seuil personnalisé
            predicted_tags = [tag for tag, prob in proba_dict.items() if prob >= threshold]
        else:
            # Utiliser la prédiction par défaut du modèle
            pred = model.predict([question])[0]
            predicted_tags = [model.mlb.classes_[i] for i, val in enumerate(pred) if val == 1]
        
        # Trier par probabilité décroissante
        predicted_tags = sorted(predicted_tags, key=lambda x: proba_dict[x], reverse=True)
        
        # Limiter au top-k si spécifié
        if top_k is not None and top_k > 0:
            predicted_tags = predicted_tags[:top_k]
        
        # Filtrer les probabilités pour ne garder que les tags prédits
        filtered_proba = {tag: proba_dict[tag] for tag in predicted_tags}
        
        predictions.append({
            'tags': predicted_tags,
            'probabilities': filtered_proba
        })
    
    return predictions


if __name__ == "__main__":
    print("Module train_model.py - Fonctions disponibles:")
    print("- train_tfidf_logreg(): Entraîne TF-IDF + LogReg")
    print("- train_tfidf_minilm(): Entraîne TF-IDF + MiniLM + LogReg")
    print("- compare_models(): Compare les deux modèles")
    print("- predict_questions(): Prédit les tags pour de nouvelles questions")