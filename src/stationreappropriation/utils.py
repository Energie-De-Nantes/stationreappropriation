import os
from pathlib import Path
from datetime import date
from dateutil.relativedelta import relativedelta
from calendar import monthrange
from dotenv import load_dotenv

def get_consumption_names() -> list[str]:
    """
    Retourne une liste des noms de consommation utilisés dans le système.

    Returns:
        list[str]: Liste des noms de consommation.
    """
    return ['HPH', 'HPB', 'HCH', 'HCB', 'HP', 'HC', 'BASE']

def check_required(config: dict[str, str], required: list[str]):
    for r in required:
        if r not in config.keys():
            raise ValueError(f'Required parameter {r} not found in {config.keys()} from .env file.')
    return config

def load_prefixed_dotenv(prefix: str='EOB_', required: list[str]=[], env_dir: str='~/station_reappropriation') -> dict[str, str]:
    # Expand the user directory and create a Path object
    env_path = Path(env_dir).expanduser() / '.env'
    if not env_path.exists():
        raise FileNotFoundError(f'No .env file found at {env_path}')
    
    # Load the .env file from the specified directory
    load_dotenv(dotenv_path=env_path)

    # Retrieve all environment variables
    env_variables = dict(os.environ)
    
    return check_required({k.replace(prefix, ''): v for k, v in env_variables.items() if k.startswith(prefix)}, required)

def gen_dates(current: date | None=None) -> tuple[date, date]:
    if not current:
        current = date.today()
    
    if current.month == 1:
        current = current.replace(month=12, year=current.year-1)
    else:
        current = current.replace(month=current.month-1)

    starting_date = current.replace(day=1)
    ending_date = current.replace(day = monthrange(current.year, current.month)[1])
    return starting_date, ending_date


def gen_trimester_dates(trimester: int, current_year: int | None = None) -> tuple[date, date]:
    if not current_year:
        current_year = date.today().year
    
    if trimester not in [1, 2, 3, 4]:
        raise ValueError("Trimester must be 1, 2, 3, or 4")

    start_month = (trimester - 1) * 3 + 1
    end_month = start_month + 2

    starting_date = date(current_year, start_month, 1)
    ending_date = date(current_year, end_month, monthrange(current_year, end_month)[1])

    return starting_date, ending_date
def gen_previous_month_boundaries(current: date | None = None) -> tuple[date, date]:
    if current is None:
        current = date.today()
    
    # Move to the first day of the previous month
    previous_month = current.replace(day=1) - relativedelta(months=1)
    
    # Use gen_month_boundaries to get the start and end dates
    return gen_month_boundaries(previous_month)

def gen_month_boundaries(current: date | None = None) -> tuple[date, date]:
    if current is None:
        current = date.today()
    starting_date = current.replace(day=1)
    ending_date = current.replace(day=monthrange(current.year, current.month)[1])
    return starting_date, ending_date

def gen_last_months(n:int=4) -> dict[str, tuple[date, date]]:
    today = date.today()
    dates = []
    for i in range(n):
        year = today.year if (today.month - i - 1) > 0 else today.year - 1
        month = ((today.month - i - 1) % 12) or 12
        # Get the last day of the month
        _, last_day = monthrange(year, month)
        # Use min to ensure we don't exceed the last day of the month
        day = min(today.day, last_day)
        dates.append(date(year, month, day))
    return {f'{d.year}_{d.month:02}':gen_month_boundaries(d) for d in dates}