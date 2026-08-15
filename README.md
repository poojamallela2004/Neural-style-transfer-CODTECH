# Neural-style-transfer-CODTECH
This project implements Neural Style Transfer, a deep learning technique that combines the content of one image with the artistic style of another image. The project uses a pre-trained VGG19 model to extract content and style features from the input images.

The generated image is optimized using content loss and style loss, with the Adam optimizer updating the image over multiple iterations. Gram matrices are used to capture the style information from different VGG19 layers.

Technologies Used:

Python
TensorFlow / Keras
VGG19
NumPy
Matplotlib
Google Colab

Key Features:

Extracts deep content and style features using VGG19.
Uses Gram matrices for style representation.
Combines content and style losses to generate the final image.
Optimizes the generated image using Adam.
Displays the original content, style, and generated images.
