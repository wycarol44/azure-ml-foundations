from inference import predict

images = [
    "sample_images/cat.jpg",
    "sample_images/dog.jpg"
]

for image_path in images:

    print(f"\nTesting: {image_path}")

    result = predict(image_path)

    print(result)