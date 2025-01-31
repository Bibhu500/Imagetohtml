import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
import numpy as np
import pickle

def load_trained_model(model_path):
    """
    Load the pre-trained TensorFlow model.
    """
    return tf.keras.models.load_model(model_path)

def load_tokenizer(tokenizer_path):
    """
    Load the tokenizer used during training.
    """
    with open(tokenizer_path, "rb") as f:
        return pickle.load(f)

def generate_html_css(model, tokenizer, image_path, max_length=100):
    """
    Generate HTML and CSS code for a given image.
    """
    image = Image.open(image_path).resize((128, 128))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    
    html_pred, css_pred = model.predict(image_array)
    
    html_sequence = np.argmax(html_pred, axis=-1).flatten()
    css_sequence = np.argmax(css_pred, axis=-1).flatten()
    
    html_code = "".join([tokenizer.index_word.get(idx, "") for idx in html_sequence if idx != 0])
    css_code = "".join([tokenizer.index_word.get(idx, "") for idx in css_sequence if idx != 0])
    
    return html_code.strip(), css_code.strip()

# Example usage
model = load_trained_model("html_css_model.h5")
tokenizer = load_tokenizer("tokenizer.pkl")
html, css = generate_html_css(model, tokenizer, "example_image.png")

print("Generated HTML:")
print(html)
print("Generated CSS:")
print(css)
