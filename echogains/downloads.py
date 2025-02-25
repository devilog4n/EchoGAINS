import echogains.CONST as CONST
import os
import requests
from tqdm import tqdm


def download_file(url, filepath,verbose=True):
    '''
    Download the file from the given URL to the given filepath
    :param url: str
        The URL to download the file from
    :param filepath: str
        The filepath where the file will be saved
    :param verbose: str
        If True, print info and progress to the standard output
    :return: None
    '''
    response = requests.get(url)
    if response.status_code == 200:
        with open(filepath, 'wb') as f:
            f.write(response.content)
        if verbose:
            print(f"Downloaded: {filepath}")
    else:
        if verbose:
            print(f"Failed to download: {url}")

def download_data_sample(download_folder,verbose=True, include_segmentation=True):
    '''
    Download sample data of 1.7GB to the target directory
    The sample data contains .npy and .mp4 files of ultrasound cardiac data obtained from an E95 scanner.
    The data contains left A2C, A4C and ALAX, PSAX and PLAX views.
    The data is stored in a folder called 'sample_data' in the target directory.
    If include_segmentation is True, the segmentation masks will be downloaded as well in a separate folder called
    'sample_data_seg'
    :param download_folder: str
        The folder where the sample data will be downloaded to. A folder called 'sample_data' will be created in this
    :param verbose: bool
        If True, print info and progress to the standard output
    :param include_segmentation: bool
        If True, download the segmentation masks as well in a separate folder called 'sample_data_seg'.
        This is only possible if custom_data_loc is None
    :return: None
    '''
    download_loc = os.path.join(download_folder, 'sample_data')
    if include_segmentation:
        download_loc_seg = os.path.join(download_folder, 'sample_data_seg')
        if not os.path.exists(download_loc_seg):
            os.makedirs(download_loc_seg)
    # Create the download directory if it doesn't exist
    if not os.path.exists(download_loc):
        os.makedirs(download_loc)

    # GitHub API URL to list contents of a directory
    api_url = CONST.DOWNLOAD_LINKS['sample_data']
    if include_segmentation:
        api_url_seg = CONST.DOWNLOAD_LINKS['sample_data_segmentations']
    if verbose:
        print(f"Fetching contents from: {api_url}")
        if include_segmentation:
            print(f"Fetching contents from: {api_url_seg}")

    urls = [api_url]
    out_locs = [download_loc]
    if include_segmentation:
        urls.append(api_url_seg)
        out_locs.append(download_loc_seg)

    for url, out_loc in zip(urls, out_locs):
        # Make a GET request to fetch the contents of the directory
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Iterate over each item in the directory
            for item in tqdm(response.json()):
                # Download only files, skip directories
                if item['type'] == 'file':
                    # Construct the download URL for the file
                    file_download_url = item['download_url']
                    # Construct the filepath where the file will be saved
                    filepath = os.path.join(out_loc, item['name'])
                    # Download the file
                    download_file(file_download_url, filepath,verbose=verbose)
        else:
            print(f"Failed to fetch contents: {url}")