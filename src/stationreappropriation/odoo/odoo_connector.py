import xmlrpc.client
import numpy as np
from typing import Any, Hashable
import pandas as pd
from pandas import DataFrame

import time
from stationreappropriation.utils import check_required

import logging
logger = logging.getLogger(__name__)

class OdooConnector:
    def __init__(self, config: dict[str, str], sim=False, url: str|None=None, db: str|None=None):

        self._config: dict[str, str] = check_required(config, ['ODOO_URL', 'ODOO_DB', 'ODOO_USERNAME', 'ODOO_PASSWORD'])
        self._url: str = config['ODOO_URL'] if url is None else url
        self._db: str = config['ODOO_DB'] if db is None else db
        self._sim: bool = sim
        self._username: str = config['ODOO_USERNAME']
        self._password: str = config['ODOO_PASSWORD']

        self._uid: int | None = None
        self._proxy: Any | None = None

    @property
    def is_connected(self) -> bool:
        return self._uid is not None and self._proxy is not None
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        logger.info(f'Disconnected from {self._db} Odoo db.')
    
    def _ensure_connection(func):
        def wrapper(self, *args, **kwargs):
            if not self.is_connected:
                self.connect()
            return func(self, *args, **kwargs)
        return wrapper
    
    def connect(self)-> None:
        self._uid = self._get_uid()
        self._proxy = xmlrpc.client.ServerProxy(f'{self._url}/xmlrpc/2/object')
        logger.info(f'Logged to {self._db}.')
    
    def disconnect(self) -> None:
        if self.is_connected:
            if hasattr(self._proxy, '_ServerProxy__transport'):
                self._proxy._ServerProxy__transport.close()

    def _get_uid(self)-> int:
        """
        Authenticates the user with the provided credentials and returns the user ID.

        Returns:
            int: The user ID obtained from the Odoo server.

        Raises:
            xmlrpc.client.Fault: If the authentication fails.

        This function creates a ServerProxy object to the Odoo server's XML-RPC interface,
        and calls the 'authenticate' method to authenticate the user with the provided credentials. 
        The user ID obtained from the Odoo server is then returned.
        """
        common_proxy = xmlrpc.client.ServerProxy(f"{self._url}/xmlrpc/2/common")
        return common_proxy.authenticate(self._db, self._username, self._password, {})
    
    @_ensure_connection
    def execute(self, model: str, method: str, args=None, kwargs=None) -> list:
        """
        Executes a method on the Odoo server.

        Args:
            model (str): The model to execute the method on.
            method (str): The method to execute.
            *args: Additional positional arguments to pass to the method.
            **kwargs: Additional keyword arguments to pass to the method.

        Returns:
            List: The result of the executed method, if it returns a list. Otherwise, a single value is wrapped in a list.

        Raises:
            xmlrpc.client.Fault: If the execution fails.

        This function creates a ServerProxy object to the Odoo server's XML-RPC interface,
        and calls the 'execute_kw' method to execute the specified method on the specified model.
        The result of the executed method is then returned, wrapped in a list if it is a single value.
        """
        if self._sim and method in ['create', 'write', 'unlink']:
            logger.info(f'Executing {method} on {model} with args {args} and kwargs {kwargs} [simulated]')
            return []
        
        if method in ['create', 'write', 'unlink']:
            time.sleep(0.5)
        
        args = args if args is not None else []
        kwargs = kwargs if kwargs is not None else {}
        logger.debug(f'Executing {method} on {model} with args {args} and kwargs {kwargs}')
        res = self._proxy.execute_kw(self._db, self._uid, self._password, model, method, args, kwargs)
        return res if isinstance(res, list) else [res]

    # medium level methods 
    def create(self, model: str, entries: list[dict[Hashable, Any]])-> list[int]:
        """
        Creates entries in the Odoo database.

        Args:
            log (Dict[str, str]): A dictionary containing the entries data.

        Returns:
            int: The ID of the newly created entries in the Odoo database.

        """
        if self._sim:
            logger.info(f'# {len(entries)} {model} creation called. [simulated]')
            return []
        
        id = self.execute(model, 'create', [entries])
        if not isinstance(id, list):
            id = [int(id)]
        logger.info(f'{model} #{id} created in Odoo db.')
        return id

    def update(self, model: str, entries: list[dict[Hashable, Any]])-> None:
        id = []
        for e in entries:
            i = int(e['id'])
            del e['id']
            data = e
            data = {k: str(v) if isinstance(v, np.str_) else v for k, v in data.items()}
            data = {k: int(v) if type(v) is np.int64 else v for k, v in data.items()}
            data = {k: float(v) if type(v) is np.int64 else v for k, v in data.items()}
            data = {k: v for k, v in data.items() if not pd.isna(v)}
            if not self._sim:
                self.execute(model, 'write', [[i], data])
            id += [i]

        logger.info(f'{len(entries)} {model} #{id} writen in Odoo db.' + ("[simulated]" if self.sim else ''))
   
    def search_read(self, model: str, filters: list[list[tuple[str, str, str]]], fields: list[str]):
        """
        Searches for entries in the Odoo database and returns them as a DataFrame.

        Args:
            model (str): The model to search in.
            filters (List[List[Tuple[str, str, str]]]): A list of filters to apply to the search.
            fields (List[str]): The fields to return for each entry.

        Returns:
            DataFrame: A DataFrame containing the search results.

        Raises:
            xmlrpc.client.Fault: If the search fails.

        This function creates a ServerProxy object to the Odoo server's XML-RPC interface,
        and calls the'search_read' method to search for the specified entries in the specified model.
        The search results are then returned as a DataFrame, with the specified fields.
        """
        resp = self.execute(model,'search_read', args=filters, kwargs={'fields': fields})
        return DataFrame(resp).rename(columns={'id': f'{model}_id'})

    def read(self, model: str, ids: list[int], fields: list[str]) -> DataFrame:
        resp = self.execute(model, 'read', [ids], {'fields': fields})
        return DataFrame(resp).rename(columns={'id': f'{model}_id'})
    


def main():
    from stationreappropriation.utils import load_prefixed_dotenv 
    # Configuration for Odoo connection
    config = load_prefixed_dotenv(prefix='SR_')

    try:
        with OdooConnector(config) as odoo:

            valid_subscriptions = odoo.search_read('sale.order', 
                filters=[[['is_subscription', '=', True], 
                          ['is_expired', '=', False], 
                          ['state', '=', 'sale'], 
                          ['subscription_state', '=', '3_progress']]], 
                fields=['x_pdl', 'x_lisse'])
        print(valid_subscriptions)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()