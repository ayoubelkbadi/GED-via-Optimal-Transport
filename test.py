import sys
import os
import torch
import json

# Ajouter le répertoire 'src' au chemin de recherche des modules Python
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir # Puisque test.py est à la racine
src_dir = os.path.join(project_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from models import GEDGW
from utils import load_all_graphs, load_labels, get_file_paths # Import get_file_paths pour le débogage
from param_parser import parameter_parser

def main():
    args = parameter_parser()
    args.dataset = 'AIDS'
    
    # Définir explicitement le chemin absolu vers le répertoire du projet
    # args.abs_path devrait être le répertoire qui contient 'json_data', 'src', etc.
    args.abs_path = project_root + "/" # S'assurer qu'il y a un slash à la fin pour la concaténation dans utils.py
    
    print(f"Utilisation du dataset: {args.dataset}")
    print(f"Chemin absolu utilisé pour les données (args.abs_path): {args.abs_path}")

    # Débogage du chargement des graphes
    print(f"Vérification des fichiers dans {args.abs_path}json_data/{args.dataset}/train/")
    train_files = get_file_paths(f"{args.abs_path}json_data/{args.dataset}/train", "json")
    print(f"Trouvé {len(train_files)} fichiers .json dans le répertoire train.")
    if not train_files:
        print("ERREUR: Aucun fichier .json trouvé dans le répertoire d'entraînement. Vérifiez le chemin et la structure des dossiers.")
        return

    # Charger tous les graphes et features
    try:
        actual_train_num, actual_val_num, actual_test_num, graphs_list_of_dicts = load_all_graphs(args.abs_path, args.dataset)
        print(f"load_all_graphs: {actual_train_num} train, {actual_val_num} val, {actual_test_num} test. Total graphes chargés: {len(graphs_list_of_dicts)}")
        if not graphs_list_of_dicts:
            print("ERREUR: graphs_list_of_dicts est vide après load_all_graphs.")
            return
        if not isinstance(graphs_list_of_dicts[0], dict):
            print(f"ERREUR: Le premier élément de graphs_list_of_dicts n'est pas un dictionnaire. Type: {type(graphs_list_of_dicts[0])}")
            return

    except Exception as e:
        print(f"Erreur lors de load_all_graphs : {e}")
        import traceback
        traceback.print_exc()
        return
        
    # Sélectionner un graphe à tester (par exemple, le premier du jeu de test)
    # L'indexation commence après les graphes d'entraînement et de validation.
    graph_idx_to_test = actual_train_num + actual_val_num # Premier graphe de test
    
    if graph_idx_to_test >= len(graphs_list_of_dicts):
        print(f"Index de graphe {graph_idx_to_test} hors limites ({len(graphs_list_of_dicts)} graphes chargés). Utilisation du graphe 0.")
        graph_idx_to_test = 0
    
    chosen_graph_dict = graphs_list_of_dicts[graph_idx_to_test]
    print(f"Test de GEDGW sur le graphe avec gid original: {chosen_graph_dict.get('gid', 'GID non trouvé')} (index dans la liste: {graph_idx_to_test})")

    # Charger les features/labels
    number_of_labels = 0
    if args.dataset in ['AIDS']:
        global_labels, features_list_of_lists = load_labels(args.abs_path, args.dataset)
        number_of_labels = len(global_labels)
        if len(features_list_of_lists) != len(graphs_list_of_dicts):
            print(f"ERREUR: Nombre de features ({len(features_list_of_lists)}) ne correspond pas au nombre de graphes ({len(graphs_list_of_dicts)}).")
            return
    if number_of_labels == 0:
        number_of_labels = 1
        features_list_of_lists = []
        for g_dict_idx, g_dict_val in enumerate(graphs_list_of_dicts):
            if not isinstance(g_dict_val, dict) or 'n' not in g_dict_val:
                print(f"ERREUR: Élément {g_dict_idx} dans graphs_list_of_dicts n'est pas un dict ou manque la clé 'n'. Valeur: {g_dict_val}")
                return
            features_list_of_lists.append([[2.0] for _ in range(g_dict_val['n'])])

    chosen_features_list = features_list_of_lists[graph_idx_to_test]

    device = torch.device('cpu')
    edge_list_1 = chosen_graph_dict['graph']
    edge_list_1_full = edge_list_1 + [[y, x] for x, y in edge_list_1]
    edge_list_1_full = edge_list_1_full + [[x, x] for x in range(chosen_graph_dict['n'])]
    
    edge_index_1 = torch.tensor(edge_list_1_full).t().long().to(device)
    features_1 = torch.tensor(chosen_features_list).float().to(device)

    data = {
        "n1": chosen_graph_dict['n'],
        "n2": chosen_graph_dict['n'],
        "edge_index_1": edge_index_1,
        "edge_index_2": edge_index_1, 
        "features_1": features_1,
        "features_2": features_1,
    }

    print("Initialisation de GEDGW...")
    try:
        gedgw_instance = GEDGW(data, args)
        print("Calcul de la GED avec GEDGW.process()...")
        _, predicted_ged = gedgw_instance.process()

        print(f"\nGED prédite par GEDGW entre le graphe et lui-même : {predicted_ged}")
        if abs(predicted_ged) < 1e-5: # Augmentation légère de la tolérance
            print("Résultat attendu (proche de 0) : SUCCÈS !")
        else:
            print(f"Résultat inattendu. La GED devrait être proche de 0, mais est {predicted_ged}.")

    except Exception as e:
        print(f"Une erreur est survenue lors de l'exécution de GEDGW : {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()