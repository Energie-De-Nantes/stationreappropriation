import os
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

def load_prefixed_dotenv(prefix: str='EOB_', required: list[str]=[]) -> dict[str, str]:
    # Load the .env file
    load_dotenv()

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