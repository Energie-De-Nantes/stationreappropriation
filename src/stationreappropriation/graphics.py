import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import matplotlib.font_manager as font_manager
import matplotlib.cm as cm

def plot_data_merge(sources_list, reference_name):
    """
    Cette fonction crée un graphique illustrant la construction d'une matrice à partir de plusieurs sources,
    avec une colonne de référence commune, en utilisant la colormap 'Set3' et affichant les noms de colonnes en vertical.

    :param sources_list: Une liste de tuples où le premier élément est le nom de la source et le second est la liste des noms de colonnes.
    :param reference_name: Le nom de la référence commune utilisée pour la fusion des sources.
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    # Paramètres pour l'affichage
    rect_width = 0.03
    rect_height = 0.3  # Augmentation de la hauteur des rectangles pour ressembler plus à des colonnes
    y_start = 0.85
    spacing = rect_width * 2  # Espace entre les matrices équivalent à deux colonnes
    reference_width = 0.015  # Largeur de la colonne de référence
    edge_color = '#4d4d4d'  # Gris foncé pour les traits
    edge_linewidth = 1.5  # Épaisseur des traits réduite

    title_fontsize = 14  # Taille de la police des titres
    text_fontsize = 12  # Taille de la police des textes

    # Choisir une police moderne
    font_properties = font_manager.FontProperties(family='DejaVu Sans', weight='bold')

    # Utiliser la colormap 'Set3' pour générer des couleurs pour chaque source
    cmap = cm.get_cmap('Set3', len(sources_list))
    colors = [cmap(i) for i in range(len(sources_list))]

    x_start = 0.1
    source_positions = []

    # Afficher les rectangles représentant les colonnes des matrices sources côte à côte avec les noms des sources
    for i, (source_name, columns) in enumerate(sources_list):
        source_positions.append((x_start, len(columns)))

        # Ajouter la colonne de référence
        ref_rect = plt.Rectangle((x_start - reference_width, y_start), reference_width, rect_height, color='#d3d3d3', ec=edge_color, linestyle='--', linewidth=edge_linewidth)
        ax.add_patch(ref_rect)
        ax.text(x_start - reference_width / 2, y_start + rect_height / 2, reference_name, ha='center', va='center', fontsize=text_fontsize, rotation=90, fontproperties=font_properties)

        ax.text(x_start + rect_width * (len(columns) - 1) / 2, y_start + rect_height + 0.02, source_name, ha='center', va='center', fontsize=title_fontsize, fontweight='bold', color=edge_color, fontproperties=font_properties)
        for j, col in enumerate(columns):
            rect = plt.Rectangle((x_start + j * rect_width, y_start), rect_width, rect_height, color=colors[i], ec=edge_color, linewidth=edge_linewidth)
            ax.add_patch(rect)
            ax.text(x_start + j * rect_width + rect_width / 2, y_start + rect_height / 2, col, ha='center', va='center', fontsize=text_fontsize, rotation=90, fontproperties=font_properties)

        x_start += len(columns) * rect_width + spacing

    # Afficher les rectangles représentant les colonnes de la matrice finale
    y_final = 0.4
    total_source_width = x_start - spacing  # Largeur totale de toutes les sources combinées
    final_matrix_width = len([col for _, columns in sources_list for col in columns]) * rect_width
    x_start_final = 0.1 + (total_source_width - final_matrix_width) / 2  # Centrer la matrice finale

    # Ajouter la colonne de référence pour la matrice finale
    ref_rect = plt.Rectangle((x_start_final - reference_width, y_final), reference_width, rect_height, color='#d3d3d3', ec=edge_color, linestyle='--', linewidth=edge_linewidth)
    ax.add_patch(ref_rect)
    ax.text(x_start_final - reference_width / 2, y_final + rect_height / 2, reference_name, ha='center', va='center', fontsize=text_fontsize, rotation=90, fontproperties=font_properties)

    all_columns = [col for _, columns in sources_list for col in columns]
    for i, col in enumerate(all_columns):
        rect = plt.Rectangle((x_start_final + i * rect_width, y_final), rect_width, rect_height, color='#e78ac3', ec=edge_color, linewidth=edge_linewidth)
        ax.add_patch(rect)
        ax.text(x_start_final + i * rect_width + rect_width / 2, y_final + rect_height / 2, col, ha='center', va='center', fontsize=text_fontsize, rotation=90, fontproperties=font_properties)

    # Ajouter le titre "Matrice Finale"
    ax.text(x_start_final + (len(all_columns) * rect_width) / 2, 0.35, 'Matrice Finale', ha='center', va='center', fontsize=title_fontsize, fontweight='bold', color=edge_color, fontproperties=font_properties)

    # Dessiner les polygones pour représenter l'ombre suivant les tracés des lignes et des côtés des rectangles
    current_index = 0
    for (x_start_line_left, num_columns), (_, columns) in zip(source_positions, sources_list):
        x_start_line_right = x_start_line_left + rect_width * (num_columns - 1)

        polygon = Polygon([
            (x_start_line_left, y_start),  # Point gauche du premier rectangle
            (x_start_line_right + rect_width, y_start),  # Point droit du dernier rectangle
            (x_start_final + (current_index + len(columns) - 1) * rect_width + rect_width, y_final + rect_height),  # Point droit du rectangle final
            (x_start_final + current_index * rect_width, y_final + rect_height)  # Point gauche du rectangle final
        ], closed=True, color='grey', alpha=0.3)

        ax.add_patch(polygon)

        # Lignes reliant les rectangles des sources à la matrice finale
        ax.plot([x_start_line_left, x_start_final + current_index * rect_width], [y_start, y_final + rect_height], color=edge_color, linewidth=edge_linewidth)
        ax.plot([x_start_line_right + rect_width, x_start_final + (current_index + len(columns) - 1) * rect_width + rect_width], [y_start, y_final + rect_height], color=edge_color, linewidth=edge_linewidth)

        current_index += len(columns)

    # Enlever les axes
    ax.axis('off')

    # Afficher la figure
    plt.show()