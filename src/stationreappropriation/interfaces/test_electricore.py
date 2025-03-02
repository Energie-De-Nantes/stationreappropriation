import marimo

__generated_with = "0.11.12"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
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

    from stationreappropriation.utils import gen_dates
    default_start, default_end = gen_dates()
    start_date_picker = mo.ui.date(value=default_start)
    end_date_picker = mo.ui.date(value=default_end)
    mo.md(
        f"Choisis la date de début {start_date_picker} et de fin {end_date_picker}"
    )
    return (
        Path,
        date,
        default_end,
        default_start,
        end_date_picker,
        env,
        flux_path,
        gen_dates,
        load_prefixed_dotenv,
        mo,
        np,
        pd,
        process_flux,
        start_date_picker,
    )


@app.cell(hide_code=True)
def _(end_date_picker, pd, start_date_picker):
    from zoneinfo import ZoneInfo
    PARIS_TZ = ZoneInfo("Europe/Paris")
    deb = pd.to_datetime(start_date_picker.value).tz_localize(PARIS_TZ)
    fin = pd.to_datetime(end_date_picker.value).tz_localize(PARIS_TZ)
    return PARIS_TZ, ZoneInfo, deb, fin


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Chargement des Flux""")
    return


@app.cell(hide_code=True)
def _():
    # from stationreappropriation.marimo_utils import download_with_marimo_progress as _dl

    # _processed, _errors = _dl(env, ['R15', 'R151', 'C15', 'F15', 'F12'], flux_path)

    # mo.md(f"Processed #{len(_processed)} files, with #{len(_errors)} erreurs")
    return


@app.cell
def _(flux_path, process_flux):
    from electricore.inputs.flux import lire_flux_c15

    historique = lire_flux_c15(process_flux('C15', flux_path / 'C15'))
    historique
    return historique, lire_flux_c15


@app.cell
def _(mo):
    mo.md(r"""## Analyse lien Evenement_Declencheur et présence des relevés Avant/Après""")
    return


@app.cell
def _():
    # from ydata_profiling import ProfileReport
    # to_profile = process_flux('C15', flux_path / 'C15')


    # _cols = ['Avant_' + c for c in ['HP', 'HC', 'BASE', 'HCH', 'HPH', 'HPB', 'HCB']]
    # to_profile['Avant'] = to_profile[_cols].notna().any(axis=1)
    # to_profile = to_profile.drop(columns=_cols)
    # _cols = ['Après_' + c for c in ['HP', 'HC', 'BASE', 'HCH', 'HPH', 'HPB', 'HCB']]
    # to_profile['Après'] = to_profile[_cols].notna().any(axis=1)
    # to_profile['~Avant'] = (~to_profile['Avant'])
    # to_profile['~Après'] = (~to_profile['Après'])
    # to_profile['AvantAprès'] = to_profile[['Avant','Après']].all(axis=1)
    # to_profile['changement_cal_dis'] = (to_profile['Avant_Id_Calendrier_Distributeur'] != to_profile['Après_Id_Calendrier_Distributeur']) & to_profile['AvantAprès']
    # to_profile = to_profile.drop(columns=_cols)
    # to_profile = to_profile.drop(columns=['Segment_Clientele', 'Num_Depannage', 'Categorie', 'Ref_Situation_Contractuelle', 'Puissance_Souscrite', 'Formule_Tarifaire_Acheminement', 'Num_Compteur', 'Date_Derniere_Modification_FTA', 'Date_Evenement', 'Ref_Demandeur', 'Id_Affaire', 'Avant_Date_Releve', 'Après_Date_Releve', 'pdl'])

    # to_profile['Evenement_Declencheur'] = to_profile['Evenement_Declencheur'].astype('category')
    # to_profile
    # profile = ProfileReport(to_profile, title="Profiling Report")
    # to_profile
    return


@app.cell
def _():
    # Calculer la corrélation entre chaque catégorie et la colonne 'Avant'
    # correlation_results = to_profile.groupby('Evenement_Declencheur')[['Avant', 'Après','changement_cal_dis']].mean().round(2)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Afficher une heatmap de la matrice de corrélation
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(correlation_results, annot=True, linewidths=0.5, fmt=".2f", cmap='viridis')
    # plt.title("Matrice de Corrélation (Evenement_Declencheur et relevé Avant Après)")
    # plt.show()
    return plt, sns


@app.cell
def _():
    # _corr = to_profile.groupby('changement_cal_dis')[['Avant', 'Après']].mean().round(2)
    # Afficher une heatmap de la matrice de corrélation
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(_corr, annot=True, linewidths=0.5, fmt=".2f", cmap='viridis')
    # plt.title("Matrice de Corrélation (Changement des calendrier Distributeur et présence relevé Avant Après)")
    # plt.show()
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        AUTRE, MDACT, MDPRM = jamais de relevés, ni Avant, ni Après

        MES, PMES, CFNE, jamais Avant, PRESQUE toujours Après (On a rien quand c'est nature = ESTIMÉ...)

        RES, CFNS = toujours Avant, jamais Après (makes sense)

        MCF toujours Avant ET Après (Modification de la programmation du calendrier fournisseur)

        MCT souvent Avant, toujours Après (Modification de la formule tarifaire d’acheminement ou de la puissance souscrite ou du statut d’Autoconsommation Collective)

        CMAT rarement Avant, souvent Après (Changement de compteur ou de disjoncteur ou Activation du calendrier Distributeur)

        Pas encore rencontré : MDBRA (Modification de données de branchement)
        """
    )
    return


@app.cell
def _():
    # profile.to_file("c15_report.html")
    return


@app.cell
def _(mo):
    mo.md(r"""# Relevés""")
    return


@app.cell
def _(flux_path, process_flux):
    from electricore.inputs.flux import lire_flux_r151

    relevés = lire_flux_r151(process_flux('R151', flux_path / 'R151'))
    relevés
    return lire_flux_r151, relevés


@app.cell
def _(deb, fin, relevés):
    relevés[(relevés['Date_Releve'] >= deb) & (relevés['Date_Releve'] <= fin)]
    return


@app.cell
def _(pd, relevés):
    from electricore.core.relevés.fonctions import interroger_relevés

    # Liste des PDLs
    _pdls = [
        '14201736551302', '14202315332431', '14202604839677', '14203907301698', '14203907320041',
        '14204920271162', '14205209746579', '14205499161812', '14206656884789', '14206656936323',
        '14207525248357', '14207959351204', '14207959429709', '14208248858780', '14208538241115',
        '14209840681823', '14210130195771', '14210564350040', '14210998339444', '14213024543240',
        '14213458652692', '14214326934238', '14214471748387', '14214616357689', '14215774190037'
    ]

    # Date unique
    _date_releve = pd.Timestamp("2024-10-25 4:00:00", tz="Europe/Paris")

    # Création du DataFrame
    _test = pd.DataFrame({"Date_Releve": [_date_releve] * len(_pdls), "pdl": _pdls})
    réponse_relevés = interroger_relevés(requêtes=_test, relevés=relevés)
    réponse_relevés
    return interroger_relevés, réponse_relevés


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Énergies""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Base de Calcul des Energies""")
    return


@app.cell
def _(deb, fin, historique):
    from electricore.core.énergies.fonctions import préparer_base_énergies

    base = préparer_base_énergies(historique, deb, fin)
    base
    return base, préparer_base_énergies


@app.cell
def _(mo):
    mo.md(r"""## Division MCT""")
    return


@app.cell
def _(deb, historique):
    from electricore.core.périmètre import extraire_modifications_impactantes

    mci = extraire_modifications_impactantes(historique=historique, deb=deb)
    mci
    return extraire_modifications_impactantes, mci


@app.cell
def _():
    # from electricore.core.énergies.fonctions import découper_périodes

    # base_calculable = découper_périodes(base, mci)
    # base_calculable
    return


@app.cell
def _():
    # modifications = mci
    # for ref_situation, modifs in modifications.groupby("Ref_Situation_Contractuelle"):
    #     print(ref_situation, modifs)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Services""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## facturation""")
    return


@app.cell
def _(deb, fin, historique, relevés):
    from electricore.core.services import facturation
    factu = facturation(deb, fin, historique, relevés)
    factu
    return factu, facturation


@app.cell
def _(deb, factu, fin):
    from electricore.core.taxes.turpe import compute_turpe, get_applicable_rules

    régles_turpe = get_applicable_rules(deb, fin)
    régles_turpe
    turpe = compute_turpe(factu, régles_turpe)
    turpe
    return compute_turpe, get_applicable_rules, régles_turpe, turpe


if __name__ == "__main__":
    app.run()
