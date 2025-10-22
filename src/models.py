"""
Module contenant les classes de modèles pour la classification multilabel.
Implémente différentes architectures : TF-IDF + LogReg, TF-IDF + MiniLM + LogReg
"""

import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sentence_transformers import SentenceTransformer
from scipy.sparse import hstack, csr_matrix


class BaseMlassifier:
    """
    Classe de base pour les modèles de classification multilabel.
    """
    
    def __init__(self, name: str = "BaseClassifier"):
        """
        Initialise le classificateur de base.
        
        Args:
            name (str): Nom du modèle
        """
        self.name = name
        self.mlb = None
        self.model = None
        self.is_fitted = False
    
    def fit(self, X, y):
        """
        Entraîne le modèle (à implémenter dans les classes enfants).
        
        Args:
            X: Features d'entraînement
            y: Labels d'entraînement
        """
        raise NotImplementedError("Méthode à implémenter dans les classes enfants")
    
    def predict(self, X):
        """
        Prédit les labels (à implémenter dans les classes enfants).
        
        Args:
            X: Features pour la prédiction
            
        Returns:
            Prédictions
        """
        raise NotImplementedError("Méthode à implémenter dans les classes enfants")
    
    def save(self, save_dir: str):
        """
        Sauvegarde le modèle (à implémenter dans les classes enfants).
        
        Args:
            save_dir (str): Répertoire de sauvegarde
        """
        raise NotImplementedError("Méthode à implémenter dans les classes enfants")
    
    def load(self, save_dir: str):
        """
        Charge le modèle (à implémenter dans les classes enfants).
        
        Args:
            save_dir (str): Répertoire contenant les modèles sauvegardés
        """
        raise NotImplementedError("Méthode à implémenter dans les classes enfants")


class TfidfLogRegClassifier(BaseMlassifier):
    """
    Classificateur multilabel utilisant TF-IDF + Logistic Regression.
    """
    
    def __init__(self, 
                 max_features: int = 30000,
                 ngram_range: Tuple[int, int] = (1, 3),
                 max_iter: int = 200,
                 min_df: int = 2,
                 max_df: float = 0.8):
        """
        Initialise le classificateur TF-IDF + LogReg.
        
        Args:
            max_features (int): Nombre maximum de features TF-IDF
            ngram_range (Tuple[int, int]): Range des n-grams (min, max)
            max_iter (int): Nombre maximum d'itérations pour LogReg
            min_df (int): Fréquence minimale de document
            max_df (float): Fréquence maximale de document (proportion)
        """
        super().__init__(name="TfidfLogReg")
        
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            min_df=min_df,
            max_df=max_df,
        )
        
        self.mlb = MultiLabelBinarizer()
        self.model = OneVsRestClassifier(LogisticRegression(max_iter=max_iter))
        self.max_iter = max_iter
    
    def fit(self, X_train, y_train):
        """
        Entraîne le modèle TF-IDF + LogReg.
        
        Args:
            X_train (pd.Series ou list): Textes d'entraînement
            y_train (list of lists): Labels d'entraînement (format multilabel)
            
        Returns:
            self: Objet entraîné
        """
        print(f"\n🔧 Entraînement du modèle {self.name}...")
        
        # Transformer les labels
        Y_train = self.mlb.fit_transform(y_train)
        print(f"📊 Classes détectées: {list(self.mlb.classes_)}")
        
        # Vectorisation TF-IDF
        print("📝 Vectorisation TF-IDF...")
        X_train_tfidf = self.tfidf.fit_transform(X_train)
        print(f"✅ Shape TF-IDF: {X_train_tfidf.shape}")
        
        # Entraînement du modèle
        print("🎯 Entraînement du classificateur...")
        self.model.fit(X_train_tfidf, Y_train)
        
        self.is_fitted = True
        print(f"✅ Modèle {self.name} entraîné avec succès")
        return self
    
    def predict(self, X_test):
        """
        Prédit les labels pour de nouvelles données.
        
        Args:
            X_test (pd.Series ou list): Textes à classifier
            
        Returns:
            np.ndarray: Prédictions binaires (shape: n_samples x n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        
        X_test_tfidf = self.tfidf.transform(X_test)
        Y_pred = self.model.predict(X_test_tfidf)
        return Y_pred
    
    def predict_proba(self, X_test):
        """
        Calcule les probabilités de prédiction.
        
        Args:
            X_test (pd.Series ou list): Textes à classifier
            
        Returns:
            np.ndarray: Probabilités (shape: n_samples x n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        
        X_test_tfidf = self.tfidf.transform(X_test)
        Y_proba = self.model.predict_proba(X_test_tfidf)
        return Y_proba
    
    def save(self, save_dir: str):
        """
        Sauvegarde le modèle et ses composants.
        
        Args:
            save_dir (str): Répertoire de sauvegarde
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        with open(f"{save_dir}/tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(self.tfidf, f)
        
        with open(f"{save_dir}/logreg_model.pkl", "wb") as f:
            pickle.dump(self.model, f)
        
        with open(f"{save_dir}/label_binarizer.pkl", "wb") as f:
            pickle.dump(self.mlb, f)
        
        print(f"💾 Modèle {self.name} sauvegardé dans {save_dir}/")
    
    def load(self, save_dir: str):
        """
        Charge un modèle sauvegardé.
        
        Args:
            save_dir (str): Répertoire contenant les modèles
        """
        with open(f"{save_dir}/tfidf_vectorizer.pkl", "rb") as f:
            self.tfidf = pickle.load(f)
        
        with open(f"{save_dir}/logreg_model.pkl", "rb") as f:
            self.model = pickle.load(f)
        
        with open(f"{save_dir}/label_binarizer.pkl", "rb") as f:
            self.mlb = pickle.load(f)
        
        self.is_fitted = True
        print(f"✅ Modèle {self.name} chargé depuis {save_dir}/")


class TfidfMiniLMClassifier(BaseMlassifier):
    """
    Classificateur multilabel utilisant TF-IDF + MiniLM embeddings + Logistic Regression.
    """
    
    def __init__(self, 
                 max_features: int = 30000,
                 ngram_range: Tuple[int, int] = (1, 3),
                 max_iter: int = 300,
                 min_df: int = 2,
                 max_df: float = 0.8,
                 embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialise le classificateur TF-IDF + MiniLM + LogReg.
        
        Args:
            max_features (int): Nombre maximum de features TF-IDF
            ngram_range (Tuple[int, int]): Range des n-grams (min, max)
            max_iter (int): Nombre maximum d'itérations pour LogReg
            min_df (int): Fréquence minimale de document
            max_df (float): Fréquence maximale de document
            embedding_model (str): Nom du modèle SentenceTransformer
        """
        super().__init__(name="TfidfMiniLM")
        
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            min_df=min_df,
            max_df=max_df,
        )
        
        self.embedding_model = SentenceTransformer(embedding_model)
        self.mlb = MultiLabelBinarizer()
        self.model = OneVsRestClassifier(LogisticRegression(max_iter=max_iter))
        self.embedding_model_name = embedding_model
    
    def fit(self, X_train, y_train):
        """
        Entraîne le modèle TF-IDF + MiniLM + LogReg.
        
        Args:
            X_train (pd.Series ou list): Textes d'entraînement
            y_train (list of lists): Labels d'entraînement
            
        Returns:
            self: Objet entraîné
        """
        print(f"\n🔧 Entraînement du modèle {self.name}...")
        
        # Transformer les labels
        Y_train = self.mlb.fit_transform(y_train)
        print(f"📊 Classes détectées: {list(self.mlb.classes_)}")
        
        # Vectorisation TF-IDF
        print("📝 Vectorisation TF-IDF...")
        X_train_tfidf = self.tfidf.fit_transform(X_train)
        print(f"✅ Shape TF-IDF: {X_train_tfidf.shape}")
        
        # Génération des embeddings
        print("🧠 Génération des embeddings MiniLM...")
        X_train_list = X_train.tolist() if hasattr(X_train, 'tolist') else list(X_train)
        X_train_emb = self.embedding_model.encode(
            X_train_list, 
            show_progress_bar=True, 
            normalize_embeddings=True
        )
        X_train_emb = np.nan_to_num(X_train_emb)
        print(f"✅ Shape embeddings: {X_train_emb.shape}")
        
        # Combinaison TF-IDF + embeddings
        print("🔗 Combinaison des features...")
        X_train_combined = hstack([X_train_tfidf, X_train_emb])
        print(f"✅ Shape combinée: {X_train_combined.shape}")
        
        # Entraînement du modèle
        print("🎯 Entraînement du classificateur...")
        self.model.fit(X_train_combined, Y_train)
        
        self.is_fitted = True
        print(f"✅ Modèle {self.name} entraîné avec succès")
        return self
    
    def predict(self, X_test):
        """
        Prédit les labels pour de nouvelles données.
        
        Args:
            X_test (pd.Series ou list): Textes à classifier
            
        Returns:
            np.ndarray: Prédictions binaires
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        
        # TF-IDF
        X_test_tfidf = self.tfidf.transform(X_test)
        
        # Embeddings
        X_test_list = X_test.tolist() if hasattr(X_test, 'tolist') else list(X_test)
        X_test_emb = self.embedding_model.encode(
            X_test_list, 
            show_progress_bar=False, 
            normalize_embeddings=True
        )
        X_test_emb = np.nan_to_num(X_test_emb)
        
        # Combinaison
        X_test_combined = hstack([X_test_tfidf, X_test_emb])
        
        Y_pred = self.model.predict(X_test_combined)
        return Y_pred
    
    def predict_proba(self, X_test):
        """
        Calcule les probabilités de prédiction.
        
        Args:
            X_test (pd.Series ou list): Textes à classifier
            
        Returns:
            np.ndarray: Probabilités
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        
        # TF-IDF
        X_test_tfidf = self.tfidf.transform(X_test)
        
        # Embeddings
        X_test_list = X_test.tolist() if hasattr(X_test, 'tolist') else list(X_test)
        X_test_emb = self.embedding_model.encode(
            X_test_list, 
            show_progress_bar=False, 
            normalize_embeddings=True
        )
        X_test_emb = np.nan_to_num(X_test_emb)
        
        # Combinaison
        X_test_combined = hstack([X_test_tfidf, X_test_emb])
        
        Y_proba = self.model.predict_proba(X_test_combined)
        return Y_proba
    
    def predict_with_threshold(self, X_test, thresholds: Dict[str, float]):
        """
        Prédit avec des seuils personnalisés par classe.
        
        Args:
            X_test (pd.Series ou list): Textes à classifier
            thresholds (Dict[str, float]): Dictionnaire {classe: seuil}
            
        Returns:
            np.ndarray: Prédictions binaires avec seuils personnalisés
        """
        Y_proba = self.predict_proba(X_test)
        Y_pred_custom = np.zeros_like(Y_proba, dtype=int)
        
        for i, cls in enumerate(self.mlb.classes_):
            threshold = thresholds.get(cls, 0.5)
            Y_pred_custom[:, i] = (Y_proba[:, i] >= threshold).astype(int)
        
        return Y_pred_custom
    
    def save(self, save_dir: str):
        """
        Sauvegarde le modèle et ses composants.
        
        Args:
            save_dir (str): Répertoire de sauvegarde
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        with open(f"{save_dir}/tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(self.tfidf, f)
        
        with open(f"{save_dir}/logreg_multilabel.pkl", "wb") as f:
            pickle.dump(self.model, f)
        
        with open(f"{save_dir}/label_binarizer.pkl", "wb") as f:
            pickle.dump(self.mlb, f)
        
        # Sauvegarder le nom du modèle d'embedding
        with open(f"{save_dir}/embedding_model_name.txt", "w") as f:
            f.write(self.embedding_model_name)
        
        print(f"💾 Modèle {self.name} sauvegardé dans {save_dir}/")
    
    def load(self, save_dir: str):
        """
        Charge un modèle sauvegardé.
        
        Args:
            save_dir (str): Répertoire contenant les modèles
        """
        with open(f"{save_dir}/tfidf_vectorizer.pkl", "rb") as f:
            self.tfidf = pickle.load(f)
        
        with open(f"{save_dir}/logreg_multilabel.pkl", "rb") as f:
            self.model = pickle.load(f)
        
        with open(f"{save_dir}/label_binarizer.pkl", "rb") as f:
            self.mlb = pickle.load(f)
        
        # Charger le modèle d'embedding
        with open(f"{save_dir}/embedding_model_name.txt", "r") as f:
            embedding_model_name = f.read().strip()
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        self.is_fitted = True
        print(f"✅ Modèle {self.name} chargé depuis {save_dir}/")


if __name__ == "__main__":
    print("Module models.py - Classes de modèles disponibles:")
    print("- TfidfLogRegClassifier: TF-IDF + Logistic Regression")
    print("- TfidfMiniLMClassifier: TF-IDF + MiniLM + Logistic Regression")