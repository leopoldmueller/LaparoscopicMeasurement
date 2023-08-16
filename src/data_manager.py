# Imports
import os
import zipfile

import requests
import synapseclient
import synapseutils

from utils import load_yaml_data


# Method for downloading hamlyn daVinci dataset
def download_davinci_dataset(download_url="http://hamlyn.doc.ic.ac.uk/vision/data/daVinci.zip",
                             local_path_to_zip_file='datasets/daVinci.zip', 
                             directory_to_extract_to = "datasets"):
    
    # Create directory if not already done
    if not os.path.exists('datasets'):
        os.makedirs('datasets')
    
    # This is a 2-factor security boolean to avoid unnecessary CO2 footprint
    if not os.path.exists(local_path_to_zip_file):

        print('Start downloading hamlyn daVinci dataset...')
        
        # Create response from url
        with requests.get(download_url, stream=True) as req:
            req.raise_for_status()

            # Initilize local file
            with open(local_path_to_zip_file, 'wb') as file:
                
                # Write content of response into local file
                for chunk in req.iter_content(chunk_size=8192):
                    file.write(chunk)

        print('...finished downloading hamlyn daVinci dataset...')
    
    else:
        print('{} already exists!'.formant(local_path_to_zip_file))

    
    print('...start to unzip hamlyn daVinci dataset...')

    # if the directory to extract to already exists, do not unzip
    if not os.path.exists(directory_to_extract_to):
        print('...start to unzip hamlyn daVinci dataset...')

        # Load zip and unzip to directory_to_extract_to directory
        with zipfile.ZipFile(local_path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(directory_to_extract_to)
        
        print('...finished to unzip hamlyn daVinci dataset!')
    
    else:
        print('{} already exists, so maybe it is already extracted!'.formant(directory_to_extract_to))
    

# Method for downloading the HEICO dataset (you need to place your login information into datasets/logfile.yaml)
def download_heico_dataset(path_to_logfile='datasets/logfile.yaml', project_id="syn21903917", local_heico_dir='datasets/HEICO'):
    
    # load your synapse login information
    logfile = load_yaml_data(path_to_logfile)
    
    # Create local directory for data set
    if not os.path.exists(local_heico_dir):
        os.makedirs(local_heico_dir)

        print("Start downloading")

        # Login to Synapse
        syn = synapseclient.login(email=logfile["synapse_account"]["email"], password=logfile["synapse_account"]["pw"], rememberMe=True)

        # Download all the files in folder files_synapse_id to a local folder
        all_files = synapseutils.syncFromSynapse(syn, entity=project_id, path=local_heico_dir)

        print("Finished downloading")

    else:
        print('{} already exists! If the data is not downloaded yet, delete the directory first!'.formant(local_heico_dir))


# Method for downloading hamlyn CholecT45 dataset (not needed for this framework)
def download_cholect45_dataset(download_url="http://lnkiy.in/cholect45dataset",
                             local_path_to_zip_file='datasets/CholecT45.zip', 
                             directory_to_extract_to = "datasets"):
    
    # Create directory if not already done
    if not os.path.exists('datasets'):
        os.makedirs('datasets')
    
    # This is a 2-factor security boolean to avoid unnecessary CO2 footprint
    if not os.path.exists(local_path_to_zip_file):

        print('Start downloading hamlyn daVinci dataset...')
        
        # Create response from url
        with requests.get(download_url, stream=True) as req:
            req.raise_for_status()

            # Initilize local file
            with open(local_path_to_zip_file, 'wb') as file:
                
                # Write content of response into local file
                for chunk in req.iter_content(chunk_size=8192):
                    file.write(chunk)

        print('...finished downloading hamlyn daVinci dataset...')
    
    else:
        print('{} already exists!'.formant(local_path_to_zip_file))

    
    print('...start to unzip CholecT45 dataset...')

    # if the directory to extract to already exists, do not unzip
    if not os.path.exists(directory_to_extract_to):
        print('...start to unzip CholecT45 dataset...')

        # Load zip and unzip to directory_to_extract_to directory
        with zipfile.ZipFile(local_path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(directory_to_extract_to)
        
        print('...finished to unzip CholecT45 dataset!')
    
    else:
        print('{} already exists, so maybe it is already extracted!'.formant(directory_to_extract_to))


if __name__ == '__main__':
    download_davinci_dataset()
    download_heico_dataset()
