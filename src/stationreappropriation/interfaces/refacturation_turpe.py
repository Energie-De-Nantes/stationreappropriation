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
def __(mo):
    mo.md("""## Délimitation temporelle""")
    return


@app.cell(hide_code=True)
def __(gen_last_months, mo):
    radio = mo.ui.radio(options=gen_last_months(), label='Choisi le Mois a traiter')
    radio
    return (radio,)


@app.cell(hide_code=True)
def __(gen_previous_month_boundaries, mo, radio):
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
def __(flux_path, process_flux):
    f12 = process_flux('F12', flux_path / 'F12')
    f12
    return (f12,)


@app.cell
def __(flux_path, process_flux):
    f15 = process_flux('F15', flux_path / 'F15')
    f15
    return (f15,)


if __name__ == "__main__":
    app.run()
