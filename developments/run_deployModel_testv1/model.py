import tensorflow as tf

def load_model():
    # Metal GPU is used automatically
    model = tf.keras.models.load_model("saved_model/model.h5")
    return model

def predict(model, input_data):
    return model.predict(input_data)