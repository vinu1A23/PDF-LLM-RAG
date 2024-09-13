import urllib.request
import shutil
...
# Download the file from `url` and save it locally under `file_name`:
url= "https://arxiv.org/pdf/1906.08172"
file_name="Mediapipe.pdf"
with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
    shutil.copyfileobj(response, out_file)
