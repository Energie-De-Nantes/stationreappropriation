import marimo

__generated_with = "0.9.14"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def __():
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
def __(env):
    from stationreappropriation.odoo import get_pdls
    pdls = get_pdls(env)
    return get_pdls, pdls


@app.cell(hide_code=True)
def __(mo):
    from stationreappropriation.utils import gen_dates
    default_start, default_end = gen_dates()
    start_date_picker = mo.ui.date(value=default_start)
    end_date_picker = mo.ui.date(value=default_end)
    mo.md(
        f"""
        ## Délimitation temporelle
        Choisis la date de début {start_date_picker} et de fin {end_date_picker}\n
        """
    )
    return (
        default_end,
        default_start,
        end_date_picker,
        gen_dates,
        start_date_picker,
    )


@app.cell(hide_code=True)
def __(
    end_date_picker,
    flux_path,
    pd,
    pdls,
    process_flux,
    start_date_picker,
):
    c15 = process_flux('C15', flux_path / 'C15')
    c15['Marque'] = c15['pdl'].isin(pdls['pdl']).apply(lambda x: 'EDN' if x else 'EL')
    _mask = (c15['Date_Evenement'] >= pd.to_datetime(start_date_picker.value)) & (c15['Date_Evenement'] <= pd.to_datetime(end_date_picker.value))
    c15_period = c15[_mask]
    c15_out_period = c15_period[(_mask) & (c15_period['Evenement_Declencheur'].isin(['RES', 'CFNS'])) & (c15_period['Marque'] == 'EDN')]
    c15_out_period
    return c15, c15_out_period, c15_period


@app.cell
def __(c15, c15_out_period):
    c15[c15['pdl'].isin(c15_out_period['pdl'])]
    return


@app.cell
def __(c15):
    c15.columns
    return


@app.cell(hide_code=True)
def __(c15, c15_out_period, pd):
    df = c15[c15['pdl'].isin(c15_out_period['pdl'])].copy()
    df['Date_Releve'] = pd.to_datetime(df['Date_Releve'])

    # Trier le dataframe par 'pdl' et 'Date_Releve' pour chaque pdl
    df_sorted = df.sort_values(by=['pdl', 'Date_Releve'])

    # Sélectionner les colonnes pour lesquelles on veut calculer la différence
    columns_to_diff = ['HPH', 'HCH', 'HPB', 'HCB', 'BASE', 'HC', 'HP']

    def calculate_diff_with_days(group):
        if len(group) >= 2:
            # Obtenir la ligne la plus récente et la plus ancienne
            most_recent = group.iloc[-1]
            oldest = group.iloc[0]
            # Calculer la différence pour les colonnes spécifiées
            values_diff = most_recent[columns_to_diff] - oldest[columns_to_diff]
            # Calculer la différence en jours entre les dates
            days_diff = (most_recent['Date_Releve'] - oldest['Date_Releve']).days + 1
            values_diff['Nb_Jours_Difference'] = days_diff
            return values_diff
        else:
            # Si un seul enregistrement, la différence est nulle
            return pd.Series([0] * len(columns_to_diff) + [0], index=columns_to_diff + ['Nb_Jours_Difference'])

    # Grouper par 'pdl' et appliquer la fonction de différence
    result = df_sorted.groupby('pdl').apply(calculate_diff_with_days).reset_index()
    result
    return calculate_diff_with_days, columns_to_diff, df, df_sorted, result


if __name__ == "__main__":
    app.run()
