import marimo

__generated_with = "0.9.30"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md(r"""# Lecture fichier(s) R63""")
    return


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import altair as alt
    import matplotlib.pyplot as plt


    file_browser = mo.ui.file_browser(
        initial_path="~/data/M023", multiple=True, filetypes=['.csv', '.json']
    )
    file_browser
    return alt, file_browser, mo, np, pd, plt


@app.cell
def __(file_browser, pd):
    _timeserie_df_list = []
    for _file in file_browser.value:
        _timeserie_df_list.append(pd.read_csv(_file.path, sep=';', parse_dates=['Horodate']))

    # Concaténer tous les dataframes de timeserie
    if _timeserie_df_list:
        timeserie_df = pd.concat(_timeserie_df_list, ignore_index=True)
    else:
        timeserie_df = pd.DataFrame()

    if not timeserie_df.empty:
        # Ajout de la Marque
        timeserie_df['pdl'] = timeserie_df['Identifiant PRM'].astype(str).str.zfill(14)

    timeserie_df
    return (timeserie_df,)


@app.cell
def __(mo):
    mo.md(r"""# Uniformisation du sampling a l'heure""")
    return


@app.cell
def __(timeserie_df):
    timeserie_h = timeserie_df[['Horodate', 'Valeur']].set_index('Horodate').resample('1H').mean()
    timeserie_h
    return (timeserie_h,)


@app.cell
def __(mo):
    mo.md(r"""# Analyse Statistique""")
    return


@app.cell
def __(timeserie_h):
    from statsmodels.tsa.seasonal import seasonal_decompose
    decomposition = seasonal_decompose(timeserie_h["Valeur"], model="additive", period=24)
    # Ajouter les composantes dans le DataFrame
    timeserie_h["Trend"] = decomposition.trend
    timeserie_h["Seasonal"] = decomposition.seasonal
    timeserie_h["Resid"] = decomposition.resid
    # Réinitialiser l'index pour Altair
    timeserie_h.reset_index(inplace=True)
    timeserie_h
    return decomposition, seasonal_decompose


@app.cell
def __(alt, mo, timeserie_h):
    # Graphique de la consommation
    consumption_chart = alt.Chart(timeserie_h).mark_bar(
    ).encode(
        x=alt.X(
            "Horodate:T",
            title="Heures",
            axis=alt.Axis(
                labelAngle=-45,
                format="%a %H:%M"  # Format pour heures/minutes en 24h
            )
        ),
        y=alt.Y("Valeur:Q", title="Consommation")
    ).properties(title="Puissance (W)", width=900, height=100)

    # Graphique de la tendance
    trend_chart = alt.Chart(timeserie_h).mark_line(color="blue").encode(
        x=alt.X("Horodate:T", title="Heures"),
        y=alt.Y("Trend:Q", title="Tendance"),
        tooltip=["Horodate:T", "Trend:Q"]
    ).properties(title="Tendance", width=900, height=100)

    # Graphique de la saisonnalité
    seasonal_chart = alt.Chart(timeserie_h).mark_line(color="orange").encode(
        x=alt.X("Horodate:T", title="Heures"),
        y=alt.Y("Seasonal:Q", title="Saisonnalité"),
        tooltip=["Horodate:T", "Seasonal:Q"]
    ).properties(title="Saisonnalité", width=900, height=100)

    # Graphique des résidus
    residual_chart = alt.Chart(timeserie_h).mark_line(color="red").encode(
        x=alt.X("Horodate:T", title="Heures"),
        y=alt.Y("Resid:Q", title="Résidus"),
        tooltip=["Horodate:T", "Resid:Q"]
    ).properties(title="Résidus", width=900, height=100)

    # Combiner les graphiques verticalement
    _final_chart = alt.vconcat(
        consumption_chart,
        trend_chart,
        seasonal_chart,
        residual_chart
    ).resolve_scale(x="shared")

    mo.ui.altair_chart(_final_chart)
    return consumption_chart, residual_chart, seasonal_chart, trend_chart


@app.cell
def __(mo):
    mo.md(r"""## Saisonnalité détaillée""")
    return


@app.cell
def __(alt, mo, timeserie_h):
    # Étape 1 : Ajouter une colonne pour l'heure
    timeserie_h["Hour"] = timeserie_h["Horodate"].dt.hour

    # Étape 2 : Calculer la saisonnalité moyenne par heure
    daily_seasonality = timeserie_h.groupby("Hour")["Seasonal"].mean().reset_index()

    # Étape 3 : Décaler l'ordre des heures pour centrer sur 5h-5h
    daily_seasonality["Shifted_Hour"] = (daily_seasonality["Hour"] - 5) % 24
    daily_seasonality = daily_seasonality.sort_values("Shifted_Hour").reset_index(drop=True)

    # Ajouter une représentation textuelle des heures réelles
    daily_seasonality["Hour_Label"] = daily_seasonality["Hour"].apply(
        lambda h: f"{h}:00"
    )

    # Étape 4 : Créer le graphique de saisonnalité journalière
    daily_seasonality_chart = alt.Chart(daily_seasonality).mark_line(color="orange").encode(
        x=alt.X(
            "Hour_Label:N",  # Utiliser des labels textuels pour afficher les heures réelles
            title="Heures (centré sur 5h-5h)",
            sort=None  # Respecter l'ordre trié
        ),
        y=alt.Y("Seasonal:Q", title="Saisonnalité"),
        tooltip=["Hour_Label:N", "Seasonal:Q"]
    ).properties(
        title="Saisonnalité journalière (centré sur 5h-5h)",
        width=900,
        height=300
    )
    mo.ui.altair_chart(daily_seasonality_chart)
    return daily_seasonality, daily_seasonality_chart


@app.cell
def __(mo):
    mo.md(r"""# Données météo""")
    return


@app.cell(hide_code=True)
def __(pd, timeserie_h):
    import openmeteo_requests

    import requests_cache
    from retry_requests import retry

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
    	"latitude": 47.2172,
    	"longitude": -1.5534,
    	"hourly": ["temperature_2m", "relative_humidity_2m"],
    	"timezone": "Europe/Paris",
    	"past_days": 31,
    	"forecast_days": 0
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
    	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
    	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
    	freq = pd.Timedelta(seconds = hourly.Interval()),
    	inclusive = "left"
    )}
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m

    hourly_dataframe = pd.DataFrame(data = hourly_data)
    hourly_dataframe["date"] = hourly_dataframe["date"].dt.tz_localize(None)
    # Filtrer hourly_dataframe pour qu'il corresponde à la plage de temps de timeserie_h
    hourly_dataframe_filtered = hourly_dataframe[
        (hourly_dataframe["date"] >= timeserie_h["Horodate"].min()) &
        (hourly_dataframe["date"] <= timeserie_h["Horodate"].max())
    ]
    hourly_dataframe_filtered
    return (
        cache_session,
        hourly,
        hourly_data,
        hourly_dataframe,
        hourly_dataframe_filtered,
        hourly_relative_humidity_2m,
        hourly_temperature_2m,
        openmeteo,
        openmeteo_requests,
        params,
        requests_cache,
        response,
        responses,
        retry,
        retry_session,
        url,
    )


@app.cell
def __(alt, hourly_dataframe_filtered):
    # Graphique de l'humidité relative
    humidity_chart = alt.Chart(hourly_dataframe_filtered).mark_line(color="purple").encode(
        x=alt.X("date:T", title="Heures", axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("relative_humidity_2m:Q", title="Humidité relative extérieure(%)"),
        tooltip=["date:T", "relative_humidity_2m:Q"]
    ).properties(title="Humidité relative extérieure", width=900, height=100)

    # Graphique de la température
    temperature_chart = alt.Chart(hourly_dataframe_filtered).mark_line(color="green").encode(
        x=alt.X("date:T", title="Heures"),
        y=alt.Y("temperature_2m:Q", title="Température extérieure (°C)"),
        tooltip=["date:T", "temperature_2m:Q"]
    ).properties(title="Température extérieure", width=900, height=100)
    return humidity_chart, temperature_chart


@app.cell
def __(mo):
    mo.md(r"""# Graphe complet""")
    return


@app.cell
def __(
    alt,
    consumption_chart,
    daily_seasonality_chart,
    humidity_chart,
    mo,
    residual_chart,
    seasonal_chart,
    temperature_chart,
    trend_chart,
):
    # Group 1: Charts with shared x-axis
    shared_charts = alt.vconcat(
        humidity_chart,
        temperature_chart,
        consumption_chart,
        trend_chart,
        seasonal_chart,
        residual_chart,

    ).resolve_scale(
        x="shared"  # Share the x-axis for these charts
    )

    # Group 2: Chart with independent x-axis
    independent_chart = daily_seasonality_chart

    # Combine the two groups
    final_chart = alt.vconcat(
        shared_charts,
        independent_chart
    )
    mo.ui.altair_chart(final_chart)
    return final_chart, independent_chart, shared_charts


if __name__ == "__main__":
    app.run()
