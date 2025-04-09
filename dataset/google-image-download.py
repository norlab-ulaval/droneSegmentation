from google_images_download import google_images_download

response = google_images_download.googleimagesdownload()

arguments = {"keywords": "Polar bears,baloons,Beaches", "limit": 20, "print_urls": True}
paths = response.download(arguments)
print(paths)
