import os
from pathlib import Path
from datetime import date

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