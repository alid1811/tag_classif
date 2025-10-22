"""
Module d'évaluation des modèles de classification multilabel.
Contient les fonctions pour calculer les métriques et générer des visualisations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    precision_recall_fscore_support,
    hamming_loss,
    accuracy_score
)
from fpdf import FPDF
from typing import Dict, List, Tuple, Optional


def compute_classification_report(Y_true: np.ndarray, 
                                  Y_pred: np.ndarray, 
                                  class_names: List[str],
                                  output_dict: bool = False) -> Dict:
    """
    Calcule le rapport de classification pour un problème multilabel.
    
    Args:
        Y_true (np.ndarray): Labels réels (shape: n_samples x n_classes)
        Y_pred (np.ndarray): Labels prédits (shape: n_samples x n_classes)
        class_names (List[str]): Noms des classes
        output_dict (bool): Si True, retourne un dictionnaire au lieu d'une string
        
    Returns:
        Dict ou str: Rapport de classification
    """
    report = classification_report(
        Y_true, 
        Y_pred, 
        target_names=class_names,
        output_dict=output_dict,
        zero_division=0
    )
    return report


def compute_additional_metrics(Y_true: np.ndarray, Y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcule des métriques additionnelles pour la classification multilabel.
    
    Args:
        Y_true (np.ndarray): Labels réels
        Y_pred (np.ndarray): Labels prédits
        
    Returns:
        Dict[str, float]: Dictionnaire contenant:
            - hamming_loss: Proportion d'erreurs sur l'ensemble des labels
            - exact_match_ratio: Proportion de prédictions parfaitement correctes
            - accuracy_score: Score d'exactitude subset
    """
    metrics = {
        'hamming_loss': hamming_loss(Y_true, Y_pred),
        'exact_match_ratio': accuracy_score(Y_true, Y_pred),
    }
    
    return metrics


def compute_per_class_metrics(Y_true: np.ndarray, 
                              Y_pred: np.ndarray,
                              class_names: List[str]) -> pd.DataFrame:
    """
    Calcule les métriques par classe (precision, recall, f1-score, support).
    
    Args:
        Y_true (np.ndarray): Labels réels
        Y_pred (np.ndarray): Labels prédits
        class_names (List[str]): Noms des classes
        
    Returns:
        pd.DataFrame: DataFrame avec les métriques par classe
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        Y_true, Y_pred, average=None, zero_division=0
    )
    
    df_metrics = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    
    return df_metrics


def generate_confusion_matrices(Y_true: np.ndarray, 
                                Y_pred: np.ndarray,
                                class_names: List[str],
                                output_dir: str = "confusion_matrices") -> List[str]:
    """
    Génère et sauvegarde les matrices de confusion pour chaque classe.
    
    Args:
        Y_true (np.ndarray): Labels réels
        Y_pred (np.ndarray): Labels prédits
        class_names (List[str]): Noms des classes
        output_dir (str): Répertoire de sauvegarde des images
        
    Returns:
        List[str]: Liste des chemins des images générées
    """
    os.makedirs(output_dir, exist_ok=True)
    image_paths = []
    
    for idx, label in enumerate(class_names):
        cm = confusion_matrix(Y_true[:, idx], Y_pred[:, idx])
        
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                   xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        plt.title(f"Confusion Matrix: {label}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        
        img_path = os.path.join(output_dir, f"cm_{label}.png")
        plt.savefig(img_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        image_paths.append(img_path)
    
    print(f"✅ {len(image_paths)} matrices de confusion générées dans {output_dir}/")
    return image_paths


def generate_metrics_summary_plot(metrics_df: pd.DataFrame, 
                                  output_path: str = "metrics_summary.png"):
    """
    Génère un graphique résumant les métriques par classe.
    
    Args:
        metrics_df (pd.DataFrame): DataFrame contenant les métriques par classe
        output_path (str): Chemin de sauvegarde du graphique
        
    Returns:
        str: Chemin du fichier généré
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metrics_df))
    width = 0.25
    
    ax.bar(x - width, metrics_df['Precision'], width, label='Precision', alpha=0.8)
    ax.bar(x, metrics_df['Recall'], width, label='Recall', alpha=0.8)
    ax.bar(x + width, metrics_df['F1-Score'], width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Classes')
    ax.set_ylabel('Scores')
    ax.set_title('Métriques par classe')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df['Class'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Graphique de synthèse sauvegardé: {output_path}")
    return output_path


def generate_pdf_report(report_text: str,
                       class_names: List[str],
                       additional_metrics: Dict[str, float],
                       confusion_matrices_dir: str,
                       output_path: str = "classification_report.pdf",
                       model_name: str = "Model"):
    """
    Génère un rapport PDF complet avec métriques et matrices de confusion.
    
    Args:
        report_text (str): Texte du rapport de classification
        class_names (List[str]): Noms des classes
        additional_metrics (Dict[str, float]): Métriques additionnelles
        confusion_matrices_dir (str): Répertoire contenant les matrices de confusion
        output_path (str): Chemin du fichier PDF à générer
        model_name (str): Nom du modèle pour le titre
        
    Returns:
        str: Chemin du fichier PDF généré
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Page de titre
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, f"Rapport de Classification - {model_name}", ln=True, align="C")
    pdf.ln(10)
    
    # Métriques globales
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Métriques Globales", ln=True)
    pdf.set_font("Arial", "", 11)
    
    for metric_name, metric_value in additional_metrics.items():
        metric_display = metric_name.replace('_', ' ').title()
        pdf.cell(0, 8, f"{metric_display}: {metric_value:.4f}", ln=True)
    
    pdf.ln(5)
    
    # Rapport de classification détaillé
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Rapport de Classification Détaillé", ln=True)
    pdf.set_font("Courier", "", 9)
    
    # Diviser le rapport en lignes pour éviter les débordements
    for line in report_text.split('\n'):
        # Tronquer les lignes trop longues
        if len(line) > 80:
            line = line[:77] + "..."
        pdf.cell(0, 5, line, ln=True)
    
    # Matrices de confusion
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Matrices de Confusion", ln=True, align="C")
    pdf.ln(5)
    
    for label in class_names:
        img_path = os.path.join(confusion_matrices_dir, f"cm_{label}.png")
        if os.path.exists(img_path):
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, f"Classe: {label}", ln=True)
            
            # Ajouter l'image avec une taille appropriée
            try:
                pdf.image(img_path, w=100)
                pdf.ln(5)
            except Exception as e:
                print(f"⚠️ Erreur lors de l'ajout de l'image {img_path}: {e}")
    
    # Sauvegarder le PDF
    pdf.output(output_path)
    print(f"📄 Rapport PDF généré: {output_path}")
    return output_path


def evaluate_model(model, 
                  X_test, 
                  Y_test,
                  class_names: List[str],
                  output_dir: str = "reports",
                  model_name: str = "Model") -> Dict:
    """
    Pipeline complet d'évaluation d'un modèle.
    
    Args:
        model: Modèle entraîné avec méthode predict()
        X_test: Features de test
        Y_test (np.ndarray): Labels de test
        class_names (List[str]): Noms des classes
        output_dir (str): Répertoire de sortie pour les rapports
        model_name (str): Nom du modèle
        
    Returns:
        Dict: Dictionnaire contenant toutes les métriques et chemins des fichiers générés
    """
    print(f"\n📊 Évaluation du modèle {model_name}...")
    
    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Prédictions
    print("🔮 Génération des prédictions...")
    Y_pred = model.predict(X_test)
    
    # Rapport de classification
    print("📝 Calcul des métriques...")
    report_dict = compute_classification_report(Y_test, Y_pred, class_names, output_dict=True)
    report_text = compute_classification_report(Y_test, Y_pred, class_names, output_dict=False)
    
    # Métriques additionnelles
    additional_metrics = compute_additional_metrics(Y_test, Y_pred)
    
    # Métriques par classe
    metrics_df = compute_per_class_metrics(Y_test, Y_pred, class_names)
    metrics_csv_path = os.path.join(output_dir, "metrics_per_class.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    
    # Générer le graphique de synthèse
    summary_plot_path = os.path.join(output_dir, "metrics_summary.png")
    generate_metrics_summary_plot(metrics_df, summary_plot_path)
    
    # Générer les matrices de confusion
    cm_dir = os.path.join(output_dir, "confusion_matrices")
    cm_paths = generate_confusion_matrices(Y_test, Y_pred, class_names, cm_dir)
    
    # Générer le rapport PDF
    pdf_path = os.path.join(output_dir, f"report_{model_name}.pdf")
    generate_pdf_report(
        report_text, 
        class_names, 
        additional_metrics, 
        cm_dir, 
        pdf_path, 
        model_name
    )
    
    # Afficher les résultats
    print("\n" + "="*60)
    print("RÉSULTATS DE L'ÉVALUATION")
    print("="*60)
    print(report_text)
    print("\nMétriques additionnelles:")
    for metric_name, metric_value in additional_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    print("="*60 + "\n")
    
    # Retourner toutes les informations
    results = {
        'report_dict': report_dict,
        'report_text': report_text,
        'additional_metrics': additional_metrics,
        'metrics_df': metrics_df,
        'metrics_csv_path': metrics_csv_path,
        'summary_plot_path': summary_plot_path,
        'confusion_matrices_dir': cm_dir,
        'pdf_path': pdf_path,
        'Y_pred': Y_pred
    }
    
    return results


if __name__ == "__main__":
    print("Module evaluation.py - Fonctions d'évaluation disponibles")
    print("- compute_classification_report()")
    print("- compute_additional_metrics()")
    print("- compute_per_class_metrics()")
    print("- generate_confusion_matrices()")
    print("- generate_pdf_report()")
    print("- evaluate_model() [Pipeline complet]")