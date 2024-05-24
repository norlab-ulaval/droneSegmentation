"""
Download iNaturalist images from csv file
"""
import os
import time
from tqdm import tqdm
from urllib.parse import urlparse
import requests
import pandas as pd
import gbif_dl
from pyinaturalist import get_observations
def dict_generator(data):
    for item in data:
        yield item
    yield {}

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        print(f"Invalid url {url}")
        return False
def download_images(url, im_name, folder):
    data_generator = []
    for i, url in enumerate(urls):
        if is_valid_url(url):
            data_generator.append({'url': url,
                                   'basename': im_name[i],
                                   })
        else:
            print(f"Invalid url {url}")
    stats = gbif_dl.stores.dl_async.download(data_generator, root=folder, loglevel="CRITICAL", batch_size=len(data_generator))
    wait_time = 2  # 2
    time.sleep(wait_time)

# Occurence
file_root = "/home/kamyar/Documents/iNaturalist_data/Red Maple/"
file_name = "observations-440071.csv"
file_path = file_root + file_name
df = pd.read_csv(file_path, delimiter=',')
# Print the header (column names) of the DataFrame
print('Header:', df.columns.tolist())
# Print the number of rows in the DataFrame
print(f'Number of rows: {len(df)}')

image_folder = "/home/kamyar/Documents/iNaturalist_data/Red Maple/images/"

#
# #for the first iteration, before disconnecting, comment these lines
# ####################################################################################################################
img_downloaded = [name for name in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, name))]
if len(img_downloaded) == 0:
    last_obs_index = 0
else:
    print(f'Number of images already downloaded: {len(img_downloaded)}')

    def get_prefix(string):
        return int(string.split('_')[0])

    sorted_list = sorted(img_downloaded, key=get_prefix)
    last_obs_id = int(sorted_list[-1].split('_')[0])
    print(f'Last observation id: {last_obs_id}')

    last_obs_index = df[df['id'] == last_obs_id].index[0]
    print(f'Last observation index: {last_obs_index}')
# ####################################################################################################################


urls = []
im_name = []

#for the first iteration, before disconnecting, replace last_obs_index with 0
for i in tqdm(range(last_obs_index, len(df[:]))):
    response = requests.get('https://ifconfig.me/ip')
    print(response.text.strip())
    obs_id = df.id[i]
    user_id = df.user_id[i]
    if user_id == 173584:
        continue
    response = get_observations(id=obs_id, user_id=user_id, per_page=10)
    if not response['results']:
        continue
    photo_ids = [photo['photo_id'] for photo in response['results'][0]['observation_photos']]
    wait_time = 0.1  # 1
    if pd.isnull(df['image_url'].iloc[i]):
        continue

    time.sleep(wait_time)
    for j, _ in enumerate(photo_ids):
        url = df.image_url[i].replace('medium', 'large')
        url = url.replace(str(photo_ids[0]), str(photo_ids[j]))
        urls.append(url)
    num_imgs = len(photo_ids)
    # Merge all into image name
    im_name_prefix = f"{obs_id}"
    im_name_prefix  = im_name_prefix.replace(" ", "_")
    for i in range(len(photo_ids)):
        im_name.append(f"{im_name_prefix}_{i+1}_of_{num_imgs}")
    num_total_images = len(urls)
    # Download in batch
    if num_total_images >= 14:
        download_images(urls, im_name, '/home/kamyar/Documents/iNaturalist_data/Red Maple/images/')
        # Reset
        urls = []
        im_name = []