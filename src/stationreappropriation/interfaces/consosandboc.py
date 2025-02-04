import marimo

__generated_with = "0.9.34"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np

    from datetime import date
    from pathlib import Path

    from electriflux.simple_reader import process_flux

    from stationreappropriation import load_prefixed_dotenv

    env = load_prefixed_dotenv(prefix='SR_')
    flux_path = Path('~/data/flux_enedis_v2/').expanduser()
    flux_path.mkdir(parents=True, exist_ok=True)
    return (
        Path,
        date,
        env,
        flux_path,
        load_prefixed_dotenv,
        mo,
        np,
        pd,
        process_flux,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Identification du sous périmètre""")
    return


@app.cell
def _(env, pd):
    from stationreappropriation.odoo import get_pdls

    pdls = get_pdls(env)
    _local = pd.DataFrame({
        'sale.order_id': [0],  # Exemple d'identifiant de commande
        'pdl': ['14295224261882']           # Exemple de PDL
    })

    # Ajouter la nouvelle ligne à la dataframe avec pd.concat
    pdls = pd.concat([pdls, _local], ignore_index=True)
    return get_pdls, pdls


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Délimitation temporelle""")
    return


@app.cell
def _(mo):
    from stationreappropriation.utils import gen_dates
    default_start, default_end = gen_dates()
    start_date_picker = mo.ui.date(value=default_start)
    end_date_picker = mo.ui.date(value=default_end)
    mo.md(
        f"Choisis la date de début {start_date_picker} et de fin {end_date_picker}"
    )
    return (
        default_end,
        default_start,
        end_date_picker,
        gen_dates,
        start_date_picker,
    )


@app.cell
def _(end_date_picker, pd, start_date_picker):
    from zoneinfo import ZoneInfo
    PARIS_TZ = ZoneInfo("Europe/Paris")
    deb = pd.to_datetime(start_date_picker.value).tz_localize(PARIS_TZ)
    fin = pd.to_datetime(end_date_picker.value).tz_localize(PARIS_TZ)
    return PARIS_TZ, ZoneInfo, deb, fin


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ## Chargement des Flux

        On a besoin du C15 et du R151.
        """
    )
    return


@app.cell(hide_code=True)
def _(PARIS_TZ, deb, fin, flux_path, pd, process_flux):
    c15 = process_flux('C15', flux_path / 'C15')
    c15['Date_Evenement'] = pd.to_datetime(c15['Date_Evenement'], utc=True).dt.tz_convert(PARIS_TZ)
    c15['Date_Releve'] = pd.to_datetime(c15['Date_Releve'], utc=True).dt.tz_convert(PARIS_TZ)

    _mask = (c15['Date_Evenement'] >= deb) & (c15['Date_Evenement'] <= fin)
    c15_periode = c15[_mask]
    return c15, c15_periode


@app.cell(hide_code=True)
def _(c15_periode):
    c15_periode[c15_periode['Evenement_Declencheur']
            .isin(['MCT'])]
    return


@app.cell(hide_code=True)
def _(flux_path, process_flux):
    from stationreappropriation.utils import get_consumption_names

    r151 = process_flux('R151', flux_path / 'R151')
    # r151['Date_Releve'] = pd.to_datetime(r151['Date_Releve'], utc=True).dt.tz_convert(PARIS_TZ)
    # Dans le r151, les index sont donnés en Wh, ce qui n'est pas le cas dans les autres flux, on va donc passer en kWh. On ne facture pas des fractions de Kwh dans tous les cas. 
    conso_cols = [c for c in get_consumption_names() if c in r151]
    r151[conso_cols] = (r151[conso_cols] / 1000).round().astype('Int64')
    r151['Unité'] = 'kWh'
    r151 = r151[r151['Id_Calendrier_Fournisseur'] != 'INCONNU']
    return conso_cols, get_consumption_names, r151


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ## Calcul des énergies consommées

        ### Qui Quoi Quand ?

        L'objectif est de produire, pour chaque couple (usager.e, pdl) dans notre périmètre, une période de facturation prenant en compte les événements particuliers. 

        Qui ? On cherche les couples (usager.e, pdl) qui sont dans le C15, dont le statut actuel est 'EN SERVICE' et celleux dont le statut est 'RESILIE' et dont la date de résiliation est dans la période de calcul.

        Quoi ? On doit savoir pour le début de la période, si c'est une entrée ou non.
        De manière analogue pour la fin de période, si c'est une sortie ou non. 

        Pour chaque (usager.e, pdl)
        On cherche les événements d'entrée : ['MES', 'PMES', 'CFNE'], puis on le met en début de période (date + index)
        On cherche les événements d'entrée : ['RES', 'CFNS'], puis on le met en fin de période (date + index)
        On cherche les événements du milieu : ['MCT'], on coupe la période en deux sous périodes (deb->MCT + MCT->fin)

        A l'issue de cette opération, on obtient un tableau avec
        """
    )
    return


@app.cell(hide_code=True)
def __():
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Définition des groupes de colonnes et des couleurs associées
    groups = {
        "Données contractuelles": ['pdl', 'Segment', 'Etat', 'Ref',
                                   'P', 'FTA', 'Compteur'],
        "Début de période": ['Date_deb', 'source_deb', 'HP_deb', 'HC_deb', 'BASE_deb'],
        "Fin de période": ['Date_fin', 'source_fin', 'HP_fin', 'HC_fin', 'BASE_fin']
    }

    colors = {
        "Données contractuelles": "lightblue",
        "Début de période": "lightgreen",
        "Fin de période": "lightcoral"
    }

    # Création du graphique
    fig, ax = plt.subplots(figsize=(10, 6))  # Augmentation de la hauteur
    ax.axis('off')

    # Calcul des positions des colonnes
    x_start = 0
    column_positions = []
    for group, cols in groups.items():
        for col in cols:
            column_positions.append((col, x_start, group))
            x_start += 1

    # Dessin des blocs avec couleurs et texte pivoté à 90°
    for col, x_pos, group in column_positions:
        rect = patches.Rectangle((x_pos, 0), 1, 1.5, linewidth=1, edgecolor='black', facecolor=colors[group])
        ax.add_patch(rect)
        ax.text(x_pos + 0.5, 1, col, ha='center', va='center', fontsize=9, rotation=90)

    # Ajout des exemples
    examples = [
        ("Ex1: MES", ["12345", "RES", "EN SERVICE", "Réf123", "6 kVA", "CU4", "CCB", 
                       "2024-01-01", "MES", "100", "50", "150", "", "", "", "", "", "",
                       "", "", "", "", "", "", "", "", "", "", ""]),
        ("Ex2: CFNE", ["67890", "PRO", "RÉSILIÉ", "Réf456", "9 kVA", "CU4", "CCB", 
                       "", "", "", "", "", "2024-03-01", "CFNE", "200", "100", "300", 
                       "", "", "", "", "", "", "", "", "", "", ""]),
        ("Ex3: MCT en FIN", ["11111", "RES", "EN SERVICE", "Réf101", "12 kVA", "CU4", "CCB", 
                              "", "", "", "", "", "2024-06-15", "MCT", "350", "200", "550", 
                              "", "", "", "", "", "", "", "", "", "", ""]),
        ("Ex4: MCT en DÉBUT", ["11111", "RES", "EN SERVICE", "Réf101", "15 kVA", "CU4", "CCB", 
                                "2024-07-01", "MCT", "400", "250", "650", "", "", "", "", "", "",
                                "", "", "", "", "", "", "", "", "", "", ""])
    ]

    # Ajout des exemples de lignes avec alpha réduit
    for i, (label, values) in enumerate(examples, start=1):
        for j, (col, x_pos, group) in enumerate(column_positions):
            rect = patches.Rectangle((x_pos, -i), 1, 1.5, linewidth=1, edgecolor='black', 
                                     facecolor=colors[group] if values[j] else "white", alpha=1)
            ax.add_patch(rect)
            ax.text(x_pos + 0.5, -i + 1, values[j], ha='center', va='center', fontsize=8, rotation=90)

    # Suppression des axes
    ax.set_xlim(0, len(column_positions))
    ax.set_ylim(-len(examples) - 1, 1.5)
    ax.set_xticks([])
    ax.set_yticks([])

    # Ajout de la légende des blocs
    legend_patches = [patches.Patch(color=colors[group], label=group) for group in groups]
    ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, fontsize=10)

    plt.show()
    return (
        ax,
        col,
        colors,
        cols,
        column_positions,
        examples,
        fig,
        group,
        groups,
        i,
        j,
        label,
        legend_patches,
        patches,
        plt,
        rect,
        values,
        x_pos,
        x_start,
    )


@app.cell
def _(c15, deb, fin):
    from stationreappropriation.moteur_metier.consos import qui_quoi_quand

    alors = qui_quoi_quand(deb, fin, c15)
    alors
    return alors, qui_quoi_quand


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Ajout des index normaux

        Pour toutes les cases de début et de fin qui ne sont pas concernées par un événement exceptionnel, on va chercher le relevé quotidien dans le R151
        """
    )
    return


@app.cell
def _(alors, deb, fin, r151):
    from stationreappropriation.moteur_metier.consos import ajout_R151

    indexes = ajout_R151(deb, fin, alors, r151)
    return ajout_R151, indexes


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Calcul des energies par différence d'index""")
    return


@app.cell
def _(indexes, pd):
    from stationreappropriation.moteur_metier.consos import calcul_energie

    energies = calcul_energie(indexes)
    energies['Puissance_Souscrite'] = pd.to_numeric(energies['Puissance_Souscrite'])
    energies
    return calcul_energie, energies


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ## Calcul des taxes

        """
    )
    return


@app.cell
def _(deb, energies, fin):
    from stationreappropriation.moteur_metier.turpe import get_applicable_rules, compute_turpe

    rules = get_applicable_rules(deb, fin)
    turpe = compute_turpe(entries=energies, rules=rules)
    turpe
    return compute_turpe, get_applicable_rules, rules, turpe


@app.cell
def _(pd, turpe):
    def custom_agg(df):
        agg_dict = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                agg_dict[col] = "sum"
            else:
                agg_dict[col] = "first"  # Prend la première valeur
        return agg_dict
    _to_drop = ['turpe_fixe_annuel', 'turpe_fixe_j', 'cg', 'cc', 'b', 'CS_fixe'] + [col for col in turpe.columns if col.endswith('_rule')]
    # Appliquer le groupby avec la fonction d'agrégation conditionnelle
    grouped = turpe.groupby("Ref_Situation_Contractuelle").agg(custom_agg(turpe))
    grouped = grouped.drop(columns=_to_drop).round(2)
    grouped
    return custom_agg, grouped


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Calcul "tout en un"
        """
    )
    return


@app.cell
def _(c15, deb, fin, grouped, r151):
    from stationreappropriation.moteur_metier.services import energies_et_taxes
    from pandas.testing import assert_frame_equal
    turpe2 = energies_et_taxes(deb, fin, c15, r151)
    assert_frame_equal(grouped, turpe2)
    turpe2
    return assert_frame_equal, energies_et_taxes, turpe2


if __name__ == "__main__":
    app.run()
