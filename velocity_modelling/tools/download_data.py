#!/usr/bin/env python3
"""
Script to download and extract NZCVM data from Dropbox.
This downloads the necessary data files for the New Zealand Community Velocity Model.
"""

import sys
import tarfile
import tempfile
from pathlib import Path

import requests
from tqdm import tqdm

# Import DATA_ROOT from constants
from velocity_modelling.constants import DATA_ROOT

# Dropbox URL for the data
DROPBOX_URL = "https://www.dropbox.com/scl/fi/53235uy9vmq8gdd58so4t/nzcvm_global_data.tar.gz?rlkey=0fpqa22fk6mf39iloe8s7lsa1&st=14xlz9or&dl=1"  # dl=1 forces download


def download_file(url: str, target_file: Path):
    """
    Download a file from a URL with progress indication.

    Parameters
    ----------
    url : str
        URL to download from
    target_file : Path
        Path to save the downloaded file
    """

    print(f"Downloading data from {url}")
    print("This may take some time depending on your internet connection...")

    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for HTTP errors

    # Get file size from headers if available
    total_size = int(response.headers.get("content-length", 0))

    # Create a progress bar
    progress_bar = tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading")

    # Write the file
    with open(target_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                progress_bar.update(len(chunk))

    progress_bar.close()
    print(f"Download completed. File saved to {target_file}")


def extract_tarfile(tar_file: Path, extract_path: Path):
    """
    Extract a tar.gz file to the specified path with progress indication.

    Parameters
    ----------
    tar_file : Path
        Path to the tar.gz file to extract
    extract_path : Path
        Path to extract the files to

    """
    print(f"Extracting files to {extract_path}...")

    with tarfile.open(tar_file, "r:gz") as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="Extracting", unit="file"):
            tar.extract(member, path=extract_path)

    print("Extraction completed.")


def main():
    """
    Download and extract the NZCVM data from Dropbox.
    """
    # Make sure DATA_ROOT exists
    data_root_path = Path(DATA_ROOT)
    data_root_path.mkdir(parents=True, exist_ok=True)

    # Create a temporary file to download to
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        temp_file_path = Path(tmp.name)

    try:
        # Download the file
        download_file(DROPBOX_URL, temp_file_path)

        # Extract the file to DATA_ROOT
        extract_tarfile(temp_file_path, data_root_path)

        print(
            f"\nData files have been successfully downloaded and extracted to {data_root_path}"
        )
        print("You can now use the NZCVM software with the complete dataset.")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        sys.exit(1)
    except tarfile.TarError as e:
        print(f"Error extracting file: {e}")
        sys.exit(1)
    except (OSError, IOError) as e:
        print(f"File system error: {e}")
        sys.exit(1)
    except PermissionError as e:
        print(f"Permission error: {e}")
        sys.exit(1)
    finally:
        # Clean up the temporary file
        if temp_file_path.exists():
            temp_file_path.unlink()


if __name__ == "__main__":
    main()
