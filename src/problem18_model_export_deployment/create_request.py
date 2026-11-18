import base64
import json

image_path = "sample_images/cat.jpg"

with open(image_path, "rb") as f:
    image_base64 = base64.b64encode(
        f.read()
    ).decode("utf-8")

request = {
    "image_base64": image_base64
}

with open("request.json", "w") as f:
    json.dump(request, f)

print("request.json created successfully.")