import os
import paramiko
import marimo as mo
import pandas as pd

from datetime import date
from pathlib import Path

from electriflux.utils import download_decrypt_extract

from stationreappropriation.utils import check_required

def download_with_marimo_progress(
    config: dict[str, str], 
    tasks: list[str], 
    local: Path,
    force: bool = False
) -> list[tuple[str, str]]:
    """
    Downloads, decrypts, and extracts new files from the SFTP server, skipping files that have already been processed.
    Uses a Marimo progress bar for progress tracking.

    Parameters:
    config (dict[str, str]): Configuration dictionary containing SFTP details, key, and IV.
    tasks (list[str]): List of directory types to process (e.g., ['R15', 'C15']).
    local (Path): The local root path to save extracted files.
    force (bool): If True, reprocess all files, even if they've been processed before.

    Returns:
    list[tuple[str, str]]: A list of tuples containing (zip_name, task_type) of newly processed files.
    """
    required = ['FTP_ADDRESS', 'FTP_USER', 'FTP_PASSWORD', 'AES_KEY', 'AES_IV'] + [f'FTP_{k}_DIR' for k in tasks]
    config = check_required(config, required)

    key = bytes.fromhex(config['AES_KEY'])
    iv = bytes.fromhex(config['AES_IV'])

    csv_path = local / "processed_zips.csv"
    if force and csv_path.exists():
        csv_path.unlink()

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        processed_zips = set(df['zip_name'])
    else:
        df = pd.DataFrame(columns=['zip_name', 'flux'])
        processed_zips = set()

    transport = paramiko.Transport((config['FTP_ADDRESS'], 22))
    transport.connect(username=config['FTP_USER'], password=config['FTP_PASSWORD'])
    sftp = paramiko.SFTPClient.from_transport(transport)

    newly_processed_files = []
    errors = {}

    try:
        for task_type in tasks:
            distant = '/flux_enedis/' + str(config[f'FTP_{task_type}_DIR'])
            local_dir = local.joinpath(task_type)
            local_dir.mkdir(parents=True, exist_ok=True)

            files_to_process = [f for f in sftp.listdir(distant) if f not in processed_zips]
            with mo.status.progress_bar(total=len(files_to_process),remove_on_exit=True) as bar:
                for file_name in files_to_process:
                    bar.update(
                        title=f"Processing {task_type}:",
                        subtitle=file_name)

                    remote_file_path = os.path.join(distant, file_name)
                    output_path = local_dir / file_name.replace('.zip', '')

                    success = download_decrypt_extract(sftp, remote_file_path, output_path, key, iv)

                    if success:
                        newly_processed_files.append((file_name, task_type))
                        df = pd.concat([df, pd.DataFrame({'zip_name': [file_name], 'flux': [task_type]})], ignore_index=True)


    except Exception as e:
        errors[distant] = str(e)

    finally:
        sftp.close()
        transport.close()

    df.to_csv(csv_path, index=False)

    return newly_processed_files, errors