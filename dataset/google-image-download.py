# from bing_image_downloader import downloader
#
# downloader.download('Rocky outcrops', limit=400, output_dir='/home/kamyar/Documents/OTHER_CLASSES', adult_filter_off=True, force_replace=False, timeout=60, verbose=True)





from google_images_download import google_images_download

response = google_images_download.googleimagesdownload()

arguments = {"keywords":"Polar bears,baloons,Beaches","limit":20,"print_urls":True}
paths = response.download(arguments)
print(paths)