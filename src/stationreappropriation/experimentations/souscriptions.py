import marimo

__generated_with = "0.9.14"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt
    from stationreappropriation import load_prefixed_dotenv

    env = load_prefixed_dotenv(prefix='SR_')
    return alt, env, load_prefixed_dotenv, mo, np, pd


@app.cell
def __(env):
    from stationreappropriation.odoo import OdooConnector

    with OdooConnector(env) as odoo:
        orders = odoo.search_read(model='sale.order', 
                                     filters=[[('is_subscription', '=', 'True'),]],
                                     fields=['date_order', 'display_name']
                                 ).rename(columns={'date_order': 'date'})
    orders
    return OdooConnector, odoo, orders


@app.cell
def __(orders, pd):
    df = orders.copy()
    # Conversion de la colonne 'start_date' en datetime
    df['date'] = pd.to_datetime(df['date']).dt.date

    # Grouper par date et compter le nombre de souscriptions
    subscriptions_per_day = df.groupby('date').size().reset_index(name='subscriptions')

    # Afficher le dataframe résultant
    subscriptions_per_day
    return df, subscriptions_per_day


@app.cell
def __(subscriptions_per_day):
    cumsum = subscriptions_per_day.copy()
    # Générer une colonne de somme cumulative
    cumsum['cumulative_subscriptions'] = subscriptions_per_day['subscriptions'].cumsum()
    cumsum = cumsum.drop(columns=['subscriptions'])
    # Remplir les dates manquantes avec la valeur précédente et définir la fréquence
    cumsum = cumsum.set_index('date').asfreq('D').ffill().reset_index()
    cumsum
    return (cumsum,)


@app.cell
def __(alt, cumsum):
    # Création de la visualisation avec Altair
    _chart = alt.Chart(cumsum).mark_line().encode(
        x='date:T',
        y='cumulative_subscriptions:Q',
        tooltip=['date:T', 'cumulative_subscriptions:Q']
    ).properties(
        title='Cumulative Subscriptions Over Time'
    )

    # Afficher le graphique
    _chart.display()
    return


@app.cell
def __(alt, cumsum, pd):
    from darts import TimeSeries
    from darts.models import ExponentialSmoothing

    # Créer une TimeSeries Darts
    _series = TimeSeries.from_dataframe(cumsum, 'date', 'cumulative_subscriptions')

    # Initialiser et ajuster un modèle de lissage exponentiel
    _model = ExponentialSmoothing()
    _model.fit(_series)

    # Prédire pour les 90 prochains jours
    _forecast = _model.predict(360)


    # Préparer les données pour Altair
    historical_df = cumsum.rename(columns={'cumulative_subscriptions': 'value'})
    forecast_df = _forecast.pd_dataframe().reset_index().rename(columns={'cumulative_subscriptions': 'value', 'index': 'date'})

    historical_df['type'] = 'Historique'
    forecast_df['type'] = 'Prévisions'
    combined_df = pd.concat([historical_df, forecast_df])
    combined_df['date'] = pd.to_datetime(combined_df['date'])

    # Visualiser les prévisions avec Altair
    _chart = alt.Chart(combined_df).mark_line().encode(
        x='date:T',
        y='value:Q',
        color='type:N'
    ).properties(
        title='Prévisions des souscriptions (ExponentialSmoothing)'
    )

    _chart
    return (
        ExponentialSmoothing,
        TimeSeries,
        combined_df,
        forecast_df,
        historical_df,
    )


@app.cell
def __(TimeSeries, cumsum):
    #from darts.models import Theta
    from darts.models import NBEATSModel
    # Créer une TimeSeries Darts
    _series = TimeSeries.from_dataframe(cumsum, 'date', 'cumulative_subscriptions')

    # Initialiser et ajuster un modèle non linéaire (NBEATS)
    nbeats_model = NBEATSModel(input_chunk_length=30, output_chunk_length=10, n_epochs=100, random_state=42)
    nbeats_model.fit(_series)
    return NBEATSModel, nbeats_model


@app.cell
def __(cumsum, nbeats_model, pd):
    # Prédire pour les 90 prochains jours
    _forecast = nbeats_model.predict(360)

    # Préparer les données pour Altair
    _historical_df = cumsum.rename(columns={'cumulative_subscriptions': 'value'})
    _forecast_df = _forecast.pd_dataframe().reset_index().rename(columns={'cumulative_subscriptions': 'value', 'index': 'date'})

    _historical_df['type'] = 'Historique'
    _forecast_df['type'] = 'Prévisions'
    nbeats_df = pd.concat([_historical_df, _forecast_df])
    nbeats_df['date'] = pd.to_datetime(nbeats_df['date'])
    return (nbeats_df,)


@app.cell(hide_code=True)
def __(alt, mo, nbeats_df):
    import datetime

    start_date = datetime.datetime(2024, 10, 1)

    _to_plot = nbeats_df[nbeats_df['date'] >= start_date]

    # Create the base chart with the date range slider selection using add_params
    data_chart = alt.Chart(_to_plot).mark_line().encode(
            x=alt.X('date:T', axis=alt.Axis(format='%b', title='Mois', tickCount='month')),
            y='value:Q',
            color='type:N'
        ).properties(
            title='Prévisions des souscriptions avec Darts (NBEATS)'
        ).interactive()

    nbeats_chart = mo.ui.altair_chart(data_chart)
    nbeats_chart
    return data_chart, datetime, nbeats_chart, start_date


@app.cell
def __(nbeats_df):
    nbeats_df
    return


@app.cell
def __(alt, cumsum, data_chart, mo, nbeats_df, pd):
    # Initialiser la valeur de référence avec la première valeur des souscriptions
    valeur_reference = cumsum['cumulative_subscriptions'].iloc[-1]
    print(valeur_reference)
    dates_doublage = []
    # Parcourir les valeurs pour identifier les moments de doublage successifs
    for idx in range(1, len(nbeats_df)):
        if nbeats_df['value'].iloc[idx] >= 2 * valeur_reference:
            # Mettre à jour la valeur de référence
            valeur_reference = nbeats_df['value'].iloc[idx]
            dates_doublage += [nbeats_df['date'].iloc[idx]]
            print(valeur_reference, nbeats_df['date'].iloc[idx])

    print(nbeats_df['value'].iloc[-1])

    _rules = alt.Chart(pd.DataFrame({'date': dates_doublage})).mark_rule(color='red', strokeDash=[5, 5],).encode(
        x='date:T'
    )

    _data_chart = data_chart + _rules

    # Afficher le graphique mis à jour
    nbeats_chart_plus = mo.ui.altair_chart(_data_chart)
    nbeats_chart_plus
    return dates_doublage, idx, nbeats_chart_plus, valeur_reference


@app.cell
def __(alt, mo, nbeats_df):
    # Créer une nouvelle colonne 'value_scenario_5' représentant une augmentation de 5%

    # Liste des pourcentages d'augmentation
    augmentations = [25, 50, 75, 100]  # en pourcentage
    #augmentations = [i for i in range(0, 150, 5)]

    # Générer des colonnes pour chaque scénario d'augmentation
    for augmentation in augmentations:
        colonne_nom = f'value_scenario_{augmentation}'
        nbeats_df[colonne_nom] = nbeats_df['value'] * (1 + augmentation / 100)

    # Empiler les données pour que les différents scénarios puissent être visualisés ensemble
    scenarios_df = nbeats_df.melt(
        id_vars=['date'], 
        value_vars=['value'] + [f'value_scenario_{augmentation}' for augmentation in augmentations], 
        var_name='type', 
        value_name='value_melted'
    )

    # Créer le graphique avec les scénarios
    _data_chart = alt.Chart(scenarios_df).mark_line().encode(
            x=alt.X('date:T', axis=alt.Axis(format='%b', title='Mois', tickCount='month')),
            y='value_melted:Q',
            color=alt.Color('type:N', legend=alt.Legend(title='Scénarios')),
            tooltip=['date:T', 'type:N', 'value_melted:Q']
        ).properties(
            title='Prévisions des souscriptions avec Scénarios d\'Augmentation'
        ).interactive()

    # Afficher le graphique avec les différents scénarios
    mo.ui.altair_chart(_data_chart)
    return augmentation, augmentations, colonne_nom, scenarios_df


@app.cell
def __(augmentations, nbeats_df, pd):
    conso_moy_MWh = 2.500
    prix_MWh = 80
    gain_spot_MWh = prix_MWh-50
    pertes_reduc_MWh = 100-prix_MWh
    df_evol = pd.DataFrame(augmentations, columns=['scenario'])

    df_evol['n_subs'] = [nbeats_df[f'value_scenario_{s}'].iloc[-1] for s in df_evol["scenario"]]
    df_evol['conso_MWh'] = df_evol['n_subs'].apply(lambda x: round(x*conso_moy_MWh))
    df_evol['conso_base_MWh'] = df_evol['conso_MWh'].iloc[0]
    df_evol['conso_bonus_MWh'] = df_evol['conso_MWh'] - df_evol['conso_base_MWh']
    df_evol['pertes_reduc_€'] = df_evol['conso_base_MWh']*pertes_reduc_MWh
    df_evol['gains_reduc_€'] = df_evol['conso_bonus_MWh']*gain_spot_MWh
    df_evol['net_€'] = df_evol['gains_reduc_€'] - df_evol['pertes_reduc_€']
    df_evol
    return conso_moy_MWh, df_evol, gain_spot_MWh, pertes_reduc_MWh, prix_MWh


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        La différence avec le scénario actuel peut s'exprimer de la manière suivante :

        \[
            \text{diff} = \underbrace{\text{conso} \times y \times (x - \overbrace{50}^{spot})}_{\text{Gains}} - \underbrace{\text{conso} \times (\overbrace{100}^{EDN} - x)}_{\text{Pertes}}, \quad y = \text{taux d'augmentation des souscriptions}
        \]

        Hypothèse : la conso est proportionnelle au nombre de souscripteurices


        Ici, notre objectif est d'exprimer $y$ en fonction de $x$ tq :
        $\text{diff}>=0$ C'est a dire que les gains soient suppérieurs aux pertes.

        On obtient donc la fonction suivante, qui donne le taux d'augmentation $y$ nécessaire pour couvrir les pertes de la réduction du prix.

        \[
        y(x) = \frac{100 - x}{x - 50}, \quad x > 50
        \]
        """
    )
    return


@app.cell(hide_code=True)
def __():
    def y(x):
        if x > 50:
            return (100 - x) / (x - 50)
        else:
            raise ValueError("La fonction n'est définie que pour x > 50.")
    return (y,)


@app.cell
def __(alt, mo, pd, y):
    # Calcul des valeurs de y pour x de 51 à 99
    x_values = list(range(70, 101))
    y_values = [y(x)*100 for x in x_values]

    # Création d'un DataFrame avec les résultats
    data = pd.DataFrame({
        'x': x_values,
        'y': y_values
    })

    # Création du graphique avec Altair
    _chart = alt.Chart(data).mark_line().encode(
        x=alt.X('x:Q', title='€/MWh'),
        y=alt.Y('y:Q', title='Augmentation % des souscription')
    ).properties(
        title="Augmentation des souscriptions nécessaire en fonction du prix du MWh"
    ).interactive()

    # Afficher le graphique
    mo.ui.altair_chart(_chart)
    return data, x_values, y_values


@app.cell
def __(np, pd):
    def generate_exponential_growth_series(start_date, end_date, initial_subscriptions, doubling_period_months=1):
        # Création d'une série de dates entre la date de début et de fin
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        days_to_double = doubling_period_months * 30  # Approximation de 30 jours par mois
        daily_growth_rate = np.power(2, 1 / days_to_double) - 1
        # Génération des souscriptions en appliquant le taux journalier
        subscriptions = initial_subscriptions * np.exp(np.log(1 + daily_growth_rate) * np.arange(len(dates)))
        # Création d'un DataFrame pour afficher les résultats
        growth_series = pd.DataFrame({'Date': dates, 'Subscriptions': subscriptions})
        return growth_series

    def generate_linear_growth_series(start_date, end_date, initial_subscriptions, daily_linear_growth):
        # Création d'une série de dates entre la date de début et de fin
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        # Calcul des souscriptions en croissance linéaire
        subscriptions = initial_subscriptions + daily_linear_growth * np.arange(len(dates))
        # Création d'un DataFrame pour afficher les résultats
        growth_series = pd.DataFrame({'Date': dates, 'Subscriptions': subscriptions})
        return growth_series
    return generate_exponential_growth_series, generate_linear_growth_series


@app.cell
def __(
    generate_exponential_growth_series,
    generate_linear_growth_series,
    pd,
):
    # Paramètres pour la série temporelle
    _start_date = '2024-10-29'
    _end_date = '2025-11-01'
    _initial_subscriptions = 142

    _l2_growth_series = generate_linear_growth_series(_start_date, _end_date, _initial_subscriptions, 2)
    _l4_growth_series = generate_linear_growth_series(_start_date, _end_date, _initial_subscriptions, 4)
    #_e1_growth_series = generate_exponential_growth_series(_start_date, _end_date, _initial_subscriptions)
    _e2_growth_series = generate_exponential_growth_series(_start_date, _end_date, _initial_subscriptions, 2)
    combined_growth_series = pd.merge(_l2_growth_series, _l4_growth_series, on='Date', suffixes=('_Lin2', '_Lin4'))
    #combined_growth_series = pd.merge(combined_growth_series, _e1_growth_series, on='Date', suffixes=('', '_Exp1'))
    combined_growth_series = pd.merge(combined_growth_series, _e2_growth_series, on='Date', suffixes=('', '_Exp2'))
    combined_growth_series = combined_growth_series.rename(columns={'Subscriptions': 'Subscriptions_Exp1'})
    combined_growth_series
    return (combined_growth_series,)


@app.cell
def __(alt, combined_growth_series, mo, pd):
    combined_growth_series_melted = combined_growth_series.melt('Date', var_name='Type', value_name='Subscriptions')
    combined_growth_series_melted

    # Définir les objectifs à atteindre (dates et valeurs)
    objectives = pd.DataFrame({
        'Date': ['2024-11-01', '2025-01-01'],
        'Label': ['Objectif 1', 'Objectif 2'],
        'Subscriptions': [150, 300]  # Valeurs cibles optionnelles pour annotations
    })

    # Tracer les demi-lignes vers les axes pour les objectifs
    _vertical_lines = alt.Chart(objectives).mark_rule(strokeDash=[5, 5], color='purple').encode(
        x='Date:T',
        y='Subscriptions:Q',
        y2=alt.datum(0)  # Lignes allant vers l'axe des abscisses
    )
    _horizontal_lines = alt.Chart(objectives).mark_rule(strokeDash=[5, 5], color='purple').encode(
        x='Date:T',
        x2=alt.datum(combined_growth_series['Date'].min()),  # Fixe l'origine des lignes sur la première date
        y='Subscriptions:Q'
    )
    # Ajouter des points d'objectifs pour plus de clarté
    _points = alt.Chart(objectives).mark_point(size=100, shape="circle", color="purple").encode(
        x='Date:T',
        y='Subscriptions:Q',
        tooltip=['Label', 'Date', 'Subscriptions']
    )
    # Créer le graphique Altair
    _chart = alt.Chart(combined_growth_series_melted).mark_line().encode(
        x='Date:T',
        y='Subscriptions:Q',
        color='Type:N'
    ).properties(
        title="Croissance des souscriptions (linéaire et exponentielle)",
        width=800,
        height=400
    ).interactive()
    mo.ui.altair_chart(_chart + _vertical_lines + _points + _horizontal_lines)
    return combined_growth_series_melted, objectives


if __name__ == "__main__":
    app.run()
