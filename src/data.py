"""
Module de preprocessing des données pour la classification multilabel.
Contient les fonctions de chargement, nettoyage et transformation des données.
"""

import os
import json
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from typing import List, Dict, Any


# Liste des tags cibles
TARGET_TAGS = [
    'math', 'graphs', 'strings', 
    'number theory', 'trees', 'geometry', 
    'games', 'probabilities'
]


def load_json_files(data_dir: str) -> pd.DataFrame:
    """
    Charge tous les fichiers JSON d'un dossier et les combine dans un DataFrame.
    
    Args:
        data_dir (str): Chemin du dossier contenant les fichiers JSON
        
    Returns:
        pd.DataFrame: DataFrame contenant tous les données chargées avec une colonne 'file_name'
        
    Raises:
        FileNotFoundError: Si le dossier n'existe pas
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Le dossier {data_dir} n'existe pas")
    
    rows = []
    
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(data_dir, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    f.seek(0)
                    
                    if first_line.startswith("{"):
                        # Format JSON normal
                        data = json.load(f)
                        data["file_name"] = file_name
                        rows.append(data)
                    else:
                        # Format JSONL (plusieurs lignes)
                        for line in f:
                            if line.strip():
                                data = json.loads(line)
                                data["file_name"] = file_name
                                rows.append(data)
            except Exception as e:
                print(f"⚠️ Erreur avec {file_name}: {e}")
    
    df = pd.DataFrame(rows)
    print(f"✅ {len(df)} exemples chargés depuis {len(rows)} fichiers")
    return df


def filter_tags(tag_list: List[str], target_tags: List[str] = TARGET_TAGS) -> List[str]:
    """
    Filtre une liste de tags pour ne garder que les tags cibles.
    
    Args:
        tag_list (List[str]): Liste de tags à filtrer
        target_tags (List[str]): Liste des tags à conserver (par défaut TARGET_TAGS)
        
    Returns:
        List[str]: Liste filtrée contenant uniquement les tags cibles
    """
    if not isinstance(tag_list, list):
        return []
    return [t for t in tag_list if t in target_tags]


def filter_dataset_by_tags(df: pd.DataFrame, tag_column: str = "tags") -> pd.DataFrame:
    """
    Filtre le dataset pour ne garder que les lignes contenant au moins un tag cible.
    
    Args:
        df (pd.DataFrame): DataFrame à filtrer
        tag_column (str): Nom de la colonne contenant les tags (par défaut "tags")
        
    Returns:
        pd.DataFrame: DataFrame filtré avec une nouvelle colonne 'raw' contenant les tags filtrés
    """
    # Créer la colonne 'raw' avec les tags filtrés
    df["raw"] = df[tag_column].apply(filter_tags)
    
    # Garder uniquement les lignes avec au moins un tag cible
    df_filtered = df[~(df["raw"].isna() | df["raw"].apply(lambda x: len(x) == 0))].copy()
    
    print(f"✅ {len(df_filtered)} exemples conservés après filtrage des tags")
    return df_filtered


def download_stopwords():
    """
    Télécharge les stopwords NLTK si nécessaire.
    
    Returns:
        None
    """
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("📥 Téléchargement des stopwords...")
        nltk.download('stopwords', quiet=True)


def clean_text(text: str, stop_words: set = None) -> str:
    """
    Nettoie un texte : mise en minuscule, suppression de caractères spéciaux et stopwords.
    
    Args:
        text (str): Texte à nettoyer
        stop_words (set): Ensemble de stopwords à supprimer (par défaut stopwords anglais)
        
    Returns:
        str: Texte nettoyé
    """
    if stop_words is None:
        download_stopwords()
        stop_words = set(stopwords.words("english"))
    
    # Mise en minuscule
    text = text.lower()
    
    # Suppression des caractères spéciaux (garder lettres, chiffres et espaces)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Normalisation des espaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Suppression des stopwords
    text = " ".join([w for w in text.split() if w not in stop_words])
    
    return text


def combine_text_columns(df: pd.DataFrame, 
                         columns: List[str] = None) -> pd.DataFrame:
    """
    Combine plusieurs colonnes textuelles en une seule colonne.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les colonnes à combiner
        columns (List[str]): Liste des noms de colonnes à combiner 
                            (par défaut: prob_desc_description, source_code, 
                             prob_desc_output_spec, prob_desc_input_spec)
        
    Returns:
        pd.DataFrame: DataFrame avec une nouvelle colonne 'combined_text'
    """
    if columns is None:
        columns = [
            'prob_desc_description',
            'source_code',
            'prob_desc_output_spec',
            'prob_desc_input_spec'
        ]
    
    # Combiner les colonnes en gérant les valeurs manquantes
    combined = df[columns[0]].fillna('')
    for col in columns[1:]:
        combined = combined + ' ' + df[col].fillna('')
    
    df['combined_text'] = combined
    print(f"✅ Colonnes combinées: {', '.join(columns)}")
    return df


def preprocess_data(df: pd.DataFrame, 
                   text_columns: List[str] = None) -> pd.DataFrame:
    """
    Pipeline complet de preprocessing : filtrage des tags, combinaison et nettoyage du texte.
    
    Args:
        df (pd.DataFrame): DataFrame brut à preprocesser
        text_columns (List[str]): Liste des colonnes à combiner (optionnel)
        
    Returns:
        pd.DataFrame: DataFrame preprocessé avec colonnes 'raw', 'combined_text' et 'clean_text'
    """
    print("\n🔧 Début du preprocessing...")
    
    # Filtrer par tags
    df = filter_dataset_by_tags(df)
    
    # Combiner les colonnes textuelles
    df = combine_text_columns(df, text_columns)
    
    # Télécharger les stopwords si nécessaire
    download_stopwords()
    stop_words = set(stopwords.words("english"))
    
    # Nettoyer le texte
    print("🧹 Nettoyage du texte...")
    df["clean_text"] = df["combined_text"].fillna("").apply(
        lambda x: clean_text(x, stop_words)
    )
    
    print(f"✅ Preprocessing terminé - {len(df)} exemples prêts")
    return df


def augment_rare_classes(X_train: pd.Series, 
                         Y_train: pd.DataFrame, 
                         rare_tags: List[str],
                         multiplier: int = 2) -> tuple:
    """
    Augmente les données pour les classes rares par duplication.
    
    Args:
        X_train (pd.Series): Textes d'entraînement
        Y_train (pd.DataFrame): Labels d'entraînement (format one-hot)
        rare_tags (List[str]): Liste des tags rares à augmenter
        multiplier (int): Facteur de multiplication (par défaut 2)
        
    Returns:
        tuple: (X_train_augmented, Y_train_augmented)
            - X_train_augmented (pd.Series): Textes augmentés
            - Y_train_augmented (np.ndarray): Labels augmentés
    """
    # Créer un DataFrame combiné
    train_df = pd.DataFrame({"text": X_train}).reset_index(drop=True)
    Y_train_reset = Y_train.reset_index(drop=True)
    train_df = pd.concat([train_df, Y_train_reset], axis=1)
    
    # Augmenter chaque classe rare
    for tag in rare_tags:
        if tag in train_df.columns:
            rare_rows = train_df[train_df[tag] == 1]
            # Dupliquer les exemples rares
            train_df = pd.concat([train_df] + [rare_rows] * (multiplier - 1), 
                                ignore_index=True)
            print(f"📈 Classe '{tag}' augmentée: {len(rare_rows)} → {len(rare_rows) * multiplier} exemples")
    
    # Séparer features et labels
    X_train_aug = train_df["text"]
    Y_train_aug = train_df[Y_train.columns].values
    
    print(f"✅ Total après augmentation: {len(X_train_aug)} exemples")
    return X_train_aug, Y_train_aug


if __name__ == "__main__":
    # Test du module
    print("Module data.py - Test des fonctions")
    
    # Test de nettoyage de texte
    sample_text = "This is a TEST with some SPECIAL characters!!! And stopwords."
    cleaned = clean_text(sample_text)
    print(f"\nTexte original: {sample_text}")
    print(f"Texte nettoyé: {cleaned}")