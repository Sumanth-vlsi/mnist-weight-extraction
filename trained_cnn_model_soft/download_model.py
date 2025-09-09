import requests

url = "https://gr-models.s3-us-west-2.amazonaws.com/mnist-model.h5"
response = requests.get(url, stream=True)

with open("mnist-model.h5", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)

print("Download complete")
