#!/usr/bin/env python3
"""
Interface en ligne de commande (CLI) pour l'entraînement et l'évaluation
des modèles de classification multilabel.
"""

import argparse
import sys
import os
import json

# Ajouter le dossier src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from train_model import train_tfidf_logreg, train_tfidf_minilm, compare_models, predict_questions


def main():
    """
    Point d'entrée principal de la CLI.
    """
    parser = argparse.ArgumentParser(
        description="CLI pour l'entraînement de modèles de classification multilabel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  # Entraîner le modèle TF-IDF + LogReg
  python cli.py train --model tfidf-logreg --data-dir ./data
  
  # Entraîner le modèle TF-IDF + MiniLM avec augmentation
  python cli.py train --model tfidf-minilm --data-dir ./data --augment
  
  # Comparer les deux modèles
  python cli.py compare --data-dir ./data
  
  # Prédire sur de nouvelles questions
  python cli.py predict --model-dir ./saved_models/tfidf_logreg --questions "What is the derivative of x^2?" "How to solve linear equations?"
  
  # Prédire depuis un fichier JSON
  python cli.py predict --model-dir ./saved_models/tfidf_minilm --input-file questions.json --output-file predictions.json
  
  # Entraîner avec paramètres personnalisés
  python cli.py train --model tfidf-logreg --data-dir ./data --test-size 0.3 --max-iter 300
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commandes disponibles')
    
    # ============================================================
    # Commande: train
    # ============================================================
    train_parser = subparsers.add_parser(
        'train',
        help='Entraîner un modèle de classification'
    )
    
    train_parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['tfidf-logreg', 'tfidf-minilm'],
        help='Type de modèle à entraîner'
    )
    
    train_parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Répertoire contenant les fichiers JSON de données'
    )
    
    train_parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion du jeu de test (par défaut: 0.2)'
    )
    
    train_parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Seed pour la reproductibilité (par défaut: 42)'
    )
    
    train_parser.add_argument(
        '--save-dir',
        type=str,
        default=None,
        help='Répertoire de sauvegarde du modèle (par défaut: saved_models/<model_name>)'
    )
    
    train_parser.add_argument(
        '--report-dir',
        type=str,
        default=None,
        help='Répertoire pour les rapports (par défaut: reports/<model_name>)'
    )
    
    # Paramètres TF-IDF
    train_parser.add_argument(
        '--max-features',
        type=int,
        default=30000,
        help='Nombre max de features TF-IDF (par défaut: 30000)'
    )
    
    train_parser.add_argument(
        '--max-iter',
        type=int,
        default=None,
        help='Nombre max d\'itérations LogReg (par défaut: 200 pour tfidf-logreg, 300 pour tfidf-minilm)'
    )
    
    # Paramètres pour TF-IDF + MiniLM
    train_parser.add_argument(
        '--augment',
        action='store_true',
        help='Activer l\'augmentation des classes rares (uniquement pour tfidf-minilm)'
    )
    
    train_parser.add_argument(
        '--rare-tags',
        type=str,
        nargs='+',
        default=None,
        help='Liste des tags rares à augmenter (par défaut: probabilities games geometry)'
    )
    
    train_parser.add_argument(
        '--augment-multiplier',
        type=int,
        default=2,
        help='Facteur de multiplication pour l\'augmentation (par défaut: 2)'
    )
    
    train_parser.add_argument(
        '--custom-thresholds',
        action='store_true',
        help='Utiliser des seuils personnalisés pour les classes rares (uniquement pour tfidf-minilm)'
    )
    
    # ============================================================
    # Commande: compare
    # ============================================================
    compare_parser = subparsers.add_parser(
        'compare',
        help='Comparer les performances de plusieurs modèles'
    )
    
    compare_parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Répertoire contenant les fichiers JSON de données'
    )
    
    compare_parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion du jeu de test (par défaut: 0.2)'
    )
    
    compare_parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Seed pour la reproductibilité (par défaut: 42)'
    )
    
    # ============================================================
    # Commande: predict
    # ============================================================
    predict_parser = subparsers.add_parser(
        'predict',
        help='Prédire les tags pour de nouvelles questions'
    )
    
    predict_parser.add_argument(
        '--model-dir',
        type=str,
        required=True,
        help='Répertoire contenant le modèle sauvegardé'
    )
    
    # Groupe mutuellement exclusif: questions directes OU fichier d'entrée
    input_group = predict_parser.add_mutually_exclusive_group(required=True)
    
    input_group.add_argument(
        '--questions',
        type=str,
        nargs='+',
        help='Une ou plusieurs questions à prédire (séparées par des espaces)'
    )
    
    input_group.add_argument(
        '--input-file',
        type=str,
        help='Fichier JSON contenant les questions (format: liste de strings ou liste de dicts avec clé "question")'
    )
    
    predict_parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Fichier JSON de sortie pour sauvegarder les prédictions (optionnel)'
    )
    
    predict_parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Seuil de probabilité pour les prédictions (par défaut: 0.5 ou seuils personnalisés du modèle)'
    )
    
    predict_parser.add_argument(
        '--top-k',
        type=int,
        default=None,
        help='Nombre maximum de tags à retourner par question (par défaut: tous les tags au-dessus du seuil)'
    )
    
    predict_parser.add_argument(
        '--verbose',
        action='store_true',
        help='Afficher les probabilités pour chaque prédiction'
    )
    
    # ============================================================
    # Parsing des arguments
    # ============================================================
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # ============================================================
    # Exécution des commandes
    # ============================================================
    if args.command == 'train':
        execute_train(args)
    elif args.command == 'compare':
        execute_compare(args)
    elif args.command == 'predict':
        execute_predict(args)


def execute_train(args):
    """
    Exécute la commande d'entraînement.
    
    Args:
        args: Arguments parsés depuis la CLI
    """
    print("\n" + "="*70)
    print(f"LANCEMENT DE L'ENTRAÎNEMENT: {args.model.upper()}")
    print("="*70)
    print(f"📂 Répertoire de données: {args.data_dir}")
    print(f"🔀 Test size: {args.test_size}")
    print(f"🎲 Random state: {args.random_state}")
    
    # Vérifier que le répertoire de données existe
    if not os.path.exists(args.data_dir):
        print(f"\n❌ Erreur: Le répertoire {args.data_dir} n'existe pas")
        sys.exit(1)
    
    # Préparer les paramètres du modèle
    model_params = {
        'max_features': args.max_features,
    }
    
    if args.max_iter is not None:
        model_params['max_iter'] = args.max_iter
    
    try:
        if args.model == 'tfidf-logreg':
            # Définir les répertoires par défaut
            save_dir = args.save_dir or "saved_models/tfidf_logreg"
            report_dir = args.report_dir or "reports/tfidf_logreg"
            
            if args.max_iter is None:
                model_params['max_iter'] = 200
            
            results = train_tfidf_logreg(
                data_dir=args.data_dir,
                test_size=args.test_size,
                random_state=args.random_state,
                save_dir=save_dir,
                report_dir=report_dir,
                **model_params
            )
            
        elif args.model == 'tfidf-minilm':
            # Définir les répertoires par défaut
            save_dir = args.save_dir or "saved_models/tfidf_minilm"
            report_dir = args.report_dir or "reports/tfidf_minilm"
            
            if args.max_iter is None:
                model_params['max_iter'] = 300
            
            rare_tags = args.rare_tags or ["probabilities", "games", "geometry"]
            
            results = train_tfidf_minilm(
                data_dir=args.data_dir,
                test_size=args.test_size,
                random_state=args.random_state,
                augment_rare=args.augment,
                rare_tags=rare_tags,
                augment_multiplier=args.augment_multiplier,
                use_custom_thresholds=args.custom_thresholds,
                save_dir=save_dir,
                report_dir=report_dir,
                **model_params
            )
        
        print("\n" + "="*70)
        print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Erreur lors de l'entraînement: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def execute_compare(args):
    """
    Exécute la commande de comparaison.
    
    Args:
        args: Arguments parsés depuis la CLI
    """
    print("\n" + "="*70)
    print("LANCEMENT DE LA COMPARAISON DES MODÈLES")
    print("="*70)
    print(f"📂 Répertoire de données: {args.data_dir}")
    
    # Vérifier que le répertoire de données existe
    if not os.path.exists(args.data_dir):
        print(f"\n❌ Erreur: Le répertoire {args.data_dir} n'existe pas")
        sys.exit(1)
    
    try:
        comparison_df = compare_models(
            data_dir=args.data_dir,
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        print("\n" + "="*70)
        print("✅ COMPARAISON TERMINÉE AVEC SUCCÈS!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Erreur lors de la comparaison: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def execute_predict(args):
    """
    Exécute la commande de prédiction.
    
    Args:
        args: Arguments parsés depuis la CLI
    """
    print("\n" + "="*70)
    print("LANCEMENT DE LA PRÉDICTION")
    print("="*70)
    print(f"📂 Répertoire du modèle: {args.model_dir}")
    
    # Vérifier que le répertoire du modèle existe
    if not os.path.exists(args.model_dir):
        print(f"\n❌ Erreur: Le répertoire {args.model_dir} n'existe pas")
        sys.exit(1)
    
    try:
        # Charger les questions
        if args.questions:
            questions = args.questions
            print(f"📝 Nombre de questions: {len(questions)}")
        else:
            # Charger depuis un fichier
            print(f"📄 Chargement depuis: {args.input_file}")
            with open(args.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Gérer différents formats
            if isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], dict):
                    # Format: [{"question": "...", ...}, ...]
                    questions = [item.get('question', item.get('text', '')) for item in data]
                else:
                    # Format: ["question1", "question2", ...]
                    questions = data
            else:
                print(f"\n❌ Erreur: Format de fichier non reconnu. Attendu: liste de strings ou liste de dicts")
                sys.exit(1)
            
            print(f"📝 Nombre de questions chargées: {len(questions)}")
        
        # Effectuer les prédictions
        print("\n🔮 Prédiction en cours...")
        predictions = predict_questions(
            questions=questions,
            model_dir=args.model_dir,
            threshold=args.threshold,
            top_k=args.top_k
        )
        
        # Afficher les résultats
        print("\n" + "="*70)
        print("RÉSULTATS DES PRÉDICTIONS")
        print("="*70)
        
        for i, (question, pred) in enumerate(zip(questions, predictions), 1):
            print(f"\n{'─'*70}")
            print(f"Question {i}: {question[:100]}{'...' if len(question) > 100 else ''}")
            print(f"{'─'*70}")
            
            tags = pred['tags']
            if tags:
                print(f"🏷️  Tags prédits: {', '.join(tags)}")
                
                if args.verbose and 'probabilities' in pred:
                    print("\n📊 Probabilités:")
                    probs = pred['probabilities']
                    for tag in tags:
                        if tag in probs:
                            print(f"   • {tag}: {probs[tag]:.4f}")
            else:
                print("🏷️  Aucun tag prédit")
        
        # Sauvegarder les résultats si demandé
        if args.output_file:
            output_data = []
            for question, pred in zip(questions, predictions):
                output_item = {
                    'question': question,
                    'predicted_tags': pred['tags']
                }
                if 'probabilities' in pred:
                    output_item['probabilities'] = pred['probabilities']
                output_data.append(output_item)
            
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Prédictions sauvegardées dans: {args.output_file}")
        
        print("\n" + "="*70)
        print("✅ PRÉDICTION TERMINÉE AVEC SUCCÈS!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Erreur lors de la prédiction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()