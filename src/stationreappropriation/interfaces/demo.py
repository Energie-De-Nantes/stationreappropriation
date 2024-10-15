import marimo

__generated_with = "0.9.9"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import pandas as pd

    from datetime import date
    from pathlib import Path

    from electriflux.simple_reader import process_flux

    from stationreappropriation import load_prefixed_dotenv

    env = load_prefixed_dotenv(prefix='ENEDIS_ODOO_BRIDGE_')
    flux_path = Path('~/data/flux_enedis_v2/').expanduser()
    flux_path.mkdir(parents=True, exist_ok=True)
    return (
        Path,
        date,
        env,
        flux_path,
        load_prefixed_dotenv,
        mo,
        pd,
        process_flux,
    )


@app.cell
def __(env, flux_path, mo):
    from stationreappropriation.marimo_utils import download_with_marimo_progress as _dl

    _processed, _errors = _dl(env, ['R15', 'R151', 'C15', 'F15', 'F12'], flux_path)

    mo.md(f"Processed #{len(_processed)} files, with #{len(_errors)} erreurs")
    return


@app.cell
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


@app.cell
def __(flux_path, process_flux):
    f15 = process_flux('F15', flux_path / 'F15')
    f15
    return (f15,)


if __name__ == "__main__":
    app.run()
