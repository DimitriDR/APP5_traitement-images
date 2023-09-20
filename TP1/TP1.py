from pathlib import Path
from typing import Dict, List

import cv2


def read_file(file_path: Path) -> Dict[int, List[str]]:
    """
    Fonction qui lit un fichier et retourne un dictionnaire indexé par les indices de lignes.
    :param file_path: Chemin du fichier à lire
    :return: Dictionnaire indexé par les indices de lignes
    """
    table = {}

    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) > 0:
                index = int(parts[0])
                data = parts[1:]
                table[index] = data

    return table


def get_fragment_size(f_index: int):
    """
    Fonction qui retourne la taille d'un fragment selon son index.
    :param f_index: Index du fragment
    :return: Taille du fragment
    """
    fragment_path = Path("assets/frag_eroded/frag_eroded_" + str(f_index) + ".png")

    opencv_frag_fd = cv2.imread(str(fragment_path), cv2.IMREAD_UNCHANGED)

    grayscale_image = cv2.cvtColor(opencv_frag_fd, cv2.COLOR_BGR2GRAY)

    # Obtenir les coordonnées du rectangle englobant l'objet non transparent.
    bbox = cv2.boundingRect(grayscale_image)

    return bbox[2] - bbox[0], bbox[3] - bbox[1]


if __name__ == '__main__':
    # On définit les constantes de tolérance
    delta_x = 1
    delta_y = 1
    delta_alpha = 1

    solution_file_path: Path = Path("assets/solution.txt")
    fragments_file_path: Path = Path("assets/fragments.txt")

    # On lit le fichier solutions.txt pour récupérer les valeurs construites puis les confronter aux valeurs réelles
    solution_file_table = read_file(solution_file_path)
    fragments_file_table = read_file(fragments_file_path)

    correct_fragment_surface: float = .0
    wrong_fragment_surface: float = .0

    # Pour chaque index dans le fichier solution.txt, on vérifie si les valeurs sont les mêmes que dans le fichier
    # fragments.txt selon la tolérance définie.
    for fragment_index in solution_file_table:

        # On compare les x, y, et alpha de chaque fragment par rapport à la solution
        comparaison_x_pos = abs(
            float(solution_file_table[fragment_index][0]) - float(fragments_file_table[fragment_index][0]))
        comparaison_y_pos = abs(
            float(solution_file_table[fragment_index][1]) - float(fragments_file_table[fragment_index][1]))
        comparaison_alpha_pos = abs(
            float(solution_file_table[fragment_index][2]) - float(fragments_file_table[fragment_index][2]))

        # On récupère la taille du fragment
        fragment_size = get_fragment_size(fragment_index)

        # Et enfin, sa surface
        fragment_surface = fragment_size[0] * fragment_size[1]

        if comparaison_x_pos <= delta_x and comparaison_y_pos <= delta_y and comparaison_alpha_pos <= delta_alpha:
            correct_fragment_surface += fragment_surface
        else:
            wrong_fragment_surface += fragment_surface

    p: float = (correct_fragment_surface - wrong_fragment_surface) / (correct_fragment_surface + wrong_fragment_surface)

    print(f"Le taux de précision est de {round(p * 100, 2)}%.")
