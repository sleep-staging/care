from bs4 import BeautifulSoup
import requests
import os

username = 'nedc'
password = 'nedc_resources'

base_dir = "/tmp/tuh/"

# specify the dir
def specify_dir(link):
    path = base_dir
    dirs = link.split('/')[-8:-1]
    for dr in dirs:
        path = os.path.join(path,dr)

        os.makedirs(path,exist_ok=True)

    return path

def RecurseLinks(base):
    response = requests.get(base,auth=(username,password))
    response_text = response.text

    print(response_text)

    soup = BeautifulSoup(response_text,'html.parser')
    for anchor in soup.find_all('a'):
        href = anchor.get('href')
        if (href.endswith('/')):
            RecurseLinks(base + href) # make recursive call w/ the new base folder
        elif ".edf" in href:
            local_path = specify_dir(os.path.join(base+href))
            os.system(f"""wget -O {os.path.join(local_path,href)} --no-check-certificate --user=nedc --password='nedc_resources' {os.path.join(base,href)}""")


# call the initial root web folder
RecurseLinks('https://isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_abnormal/v2.0.0/edf/')
