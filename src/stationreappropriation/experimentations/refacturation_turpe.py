import marimo

__generated_with = "0.11.14"
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
    from stationreappropriation.utils import gen_last_months, gen_previous_month_boundaries

    env = load_prefixed_dotenv(prefix='SR_')

    flux_path = Path('~/data/flux_enedis_v2/').expanduser()
    flux_path.mkdir(parents=True, exist_ok=True)
    return (
        Path,
        date,
        env,
        flux_path,
        gen_last_months,
        gen_previous_month_boundaries,
        load_prefixed_dotenv,
        mo,
        np,
        pd,
        process_flux,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Délimitation temporelle

        Tu peux rentrer soit le mois, soit les dates de début et de fin directement.
        """
    )
    return


@app.cell(hide_code=True)
def _(gen_last_months, mo):
    radio = mo.ui.radio(options=gen_last_months(), label='Choisi le Mois a traiter')
    radio
    return (radio,)


@app.cell(hide_code=True)
def _(gen_previous_month_boundaries, mo, radio):
    default_start, default_end = radio.value if radio.value is not None else gen_previous_month_boundaries()
    start_date_picker = mo.ui.date(value=default_start)
    end_date_picker = mo.ui.date(value=default_end)
    mo.md(
        f"""
        Choisis la date de début {start_date_picker} et de fin {end_date_picker}\n
        """
    )
    return default_end, default_start, end_date_picker, start_date_picker


@app.cell
def _(lignes_f12, lignes_f15):
    # Fusionner
    recap = lignes_f15.merge(lignes_f12, on='Taux_TVA_Applicable', how='outer', suffixes=('_f15', '_f12'))
    recap['total'] = recap[[c for c in recap.columns if c.startswith('Montant_HT')]].sum(axis=1, min_count=1)
    recap
    return (recap,)


@app.cell
def _(mo):
    mo.md(r"""# Détails""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Identification des Marques""")
    return


@app.cell
def _(env, mo, pd):
    from stationreappropriation.odoo import get_pdls
    pdls = get_pdls(env)
    _local = pd.DataFrame({
        'sale.order_id': [0, 0, 0],  # Exemple d'identifiant de commande
        'pdl': ['14295224261882', '50070117855585', '50000508594660']           # Exemple de PDL
    })

    # Ajouter la nouvelle ligne à la dataframe avec pd.concat
    pdls = pd.concat([pdls, _local], ignore_index=True)
    # mo.vstack()
    mo.md(f"""## Liste des PDLs d'EDN """)
    return get_pdls, pdls


@app.cell
def _(mo):
    mo.md(r"""## F15""")
    return


@app.cell
def _(end_date_picker, flux_path, pd, pdls, process_flux, start_date_picker):
    f15 = process_flux('F15', flux_path / 'F15')
    f15['Marque'] = f15['pdl'].isin(pdls['pdl']).apply(lambda x: 'EDN' if x else 'ZEL')
    f15['start_date'] = pd.to_datetime(start_date_picker.value)
    f15['end_date'] = pd.to_datetime(end_date_picker.value)

    f15 = f15[f15['Date_Facture'] >= f15['start_date']]
    f15 = f15[f15['Date_Facture'] <= f15['end_date']]
    f15 = f15.drop(columns=['start_date', 'end_date'])
    f15['Montant_HT'] = pd.to_numeric(f15['Montant_HT'])
    f15
    return (f15,)


@app.cell
def _(convert_tva, f15):
    f15_zel = f15[f15['Marque']=='ZEL'].assign(Taux_TVA_Applicable=f15['Taux_TVA_Applicable'].apply(convert_tva))
    # f15_zel['Taux_TVA_Applicable'] = f15_zel['Taux_TVA_Applicable'].apply(convert_tva)
    f15_zel
    return (f15_zel,)


@app.cell
def _(f15_zel):
    lignes_f15 = f15_zel[['Taux_TVA_Applicable', 'Montant_HT']].groupby('Taux_TVA_Applicable').sum()
    lignes_f15
    return (lignes_f15,)


@app.cell
def _(end_date_picker, flux_path, pd, pdls, process_flux, start_date_picker):
    f12 = process_flux('F12', flux_path / 'F12')
    f12['Marque'] = f12['pdl'].isin(pdls['pdl']).apply(lambda x: 'EDN' if x else 'ZEL')
    f12['Montant_HT'] = pd.to_numeric(f12['Montant_HT'])
    f12['start_date'] = pd.to_datetime(start_date_picker.value)
    f12['end_date'] = pd.to_datetime(end_date_picker.value)

    f12 = f12[f12['Date_Facture'] >= f12['start_date']]
    f12 = f12[f12['Date_Facture'] <= f12['end_date']]
    f12 = f12.drop(columns=['start_date', 'end_date'])
    f12
    return (f12,)


@app.cell
def _(mo):
    mo.md("""## Filtrage ZEL et Uniformisation des Taux TVA""")
    return


@app.cell
def _():
    # Convertir en float tout sauf 'NS'
    def convert_tva(value):
        try:
            return str(float(value))  # Convertit en float si possible
        except ValueError:
            return value  # Garde 'NS' inchangé
    return (convert_tva,)


@app.cell
def _(convert_tva, f12):
    f12_zel = f12[f12['Marque']=='ZEL'].assign(Taux_TVA_Applicable=f12['Taux_TVA_Applicable'].apply(convert_tva))
    f12_zel
    return (f12_zel,)


@app.cell
def _(f12_zel):
    lignes_f12 = f12_zel[['Taux_TVA_Applicable', 'Montant_HT']].groupby('Taux_TVA_Applicable').sum()
    lignes_f12
    return (lignes_f12,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
