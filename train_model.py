import tensorflow as tf
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from PIL import Image
import io
import pickle

def load_data_from_sql(sql_file):
    """
    Load HTML, CSS, and image data from a SQL file.
    """
    conn = sqlite3.connect(sql_file)
    cursor = conn.cursor()
    cursor.execute("SELECT html, css, image_blob FROM dapui_trainer_elements")
    data = cursor.fetchall()
    conn.close()

    if not data:
        raise ValueError("No data found in the database.")

    html_data, css_data, images = [], [], []
    for html, css, image_blob in data:
        html_data.append(html)
        css_data.append(css)
        image = Image.open(io.BytesIO(image_blob))
        image = image.convert("RGB")  # Ensure 3 channels (RGB)
        images.append(np.array(image.resize((128, 128))) / 255.0)  # Normalize and resize

    return html_data, css_data, np.array(images)

def preprocess_text_data(html_data, css_data):
    """
    Preprocess HTML and CSS data by tokenizing them.
    """
    tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
    tokenizer.fit_on_texts(html_data + css_data)
    
    html_sequences = tokenizer.texts_to_sequences(html_data)
    css_sequences = tokenizer.texts_to_sequences(css_data)
    
    max_len = max(max(len(seq) for seq in html_sequences), max(len(seq) for seq in css_sequences))
    html_padded = tf.keras.preprocessing.sequence.pad_sequences(html_sequences, padding="post", maxlen=max_len)
    css_padded = tf.keras.preprocessing.sequence.pad_sequences(css_sequences, padding="post", maxlen=max_len)
    
    return html_padded, css_padded, tokenizer, max_len

def build_model(input_shape, vocab_size, sequence_length):
    """
    Build a TensorFlow model that maps images to sequences of tokens (HTML and CSS).
    """
    image_input = tf.keras.layers.Input(shape=input_shape, name="image_input")
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")(image_input)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.RepeatVector(sequence_length)(x)  # Repeat for sequence length

    # LSTM layers for sequence prediction
    lstm_output = tf.keras.layers.LSTM(128, return_sequences=True)(x)
    html_output = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(vocab_size, activation="softmax"), name="html_output"
    )(lstm_output)
    css_output = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(vocab_size, activation="softmax"), name="css_output"
    )(lstm_output)

    model = tf.keras.models.Model(inputs=image_input, outputs=[html_output, css_output])
    model.compile(
        optimizer="adam",
        loss={
            "html_output": "sparse_categorical_crossentropy",
            "css_output": "sparse_categorical_crossentropy"
        },
        metrics={
            "html_output": ["accuracy"],
            "css_output": ["accuracy"]
        }
    )
    return model

def train_model(sql_file, output_model_path):
    """
    Train the model using data from the SQL file.
    """
    # Load and preprocess data
    html_data, css_data, images = load_data_from_sql(sql_file)
    html_padded, css_padded, tokenizer, max_len = preprocess_text_data(html_data, css_data)
    
    # Split data into training and validation sets
    X_train, X_val, y_html_train, y_html_val, y_css_train, y_css_val = train_test_split(
        images, html_padded, css_padded, test_size=0.2, random_state=42
    )
    
    # Build and train the model
    model = build_model(input_shape=(128, 128, 3), vocab_size=len(tokenizer.word_index) + 1, sequence_length=max_len)
    model.fit(
        X_train,
        {"html_output": y_html_train, "css_output": y_css_train},
        validation_data=(X_val, {"html_output": y_html_val, "css_output": y_css_val}),
        epochs=10,
        batch_size=32
    )
    model.save(output_model_path)
    print(f"Model saved to {output_model_path}")

# Example usage
if __name__ == "__main__":
    train_model("dap_db.db", "html_css_model.h5")
