import io
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from google.cloud import vision
from google.cloud.vision_v1 import types

# Create a Vision client
client = vision.ImageAnnotatorClient()

# Read the image file
image_path = "image.jpeg"
with io.open(image_path, "rb") as image_file:
    content = image_file.read()

# Create an instance of vision.Image
image = types.Image(content=content)

# Detect faces in the image
faces = client.face_detection(image=image)

# Load the image using PIL
img = Image.open(image_path)
draw = ImageDraw.Draw(img)

# Initialize face count
face_count = 0

# Print the results and draw bounding boxes
for face in faces.face_annotations:
    face_count += 1
    print("Face found:")
    print("-- Location:", face.bounding_poly)
    print("-- Confidence:", face.detection_confidence)
    
    # Extract bounding box vertices
    vertices = [(vertex.x, vertex.y) for vertex in face.bounding_poly.vertices]

    # Draw bounding box on the image
    draw.polygon(vertices, outline='red')

    # Display confidence level below the bounding box with larger font size
    text_position = (vertices[0][0], vertices[2][1] + 10)
    draw.text(text_position, f"Confidence: {face.detection_confidence:.2f}", fill='red', font=None, font_size=24)

# Add total face count text
font = ImageFont.truetype("arial.ttf", 20)  # Adjust font and size as needed
text_x = 10  # Position text at the top left corner
text_y = 10
draw.text((text_x, text_y), f"Total Faces Detected: {face_count}", font=font, fill='red')

# Show the image with bounding boxes and confidence levels
plt.imshow(img)
plt.axis('off')
plt.show()
