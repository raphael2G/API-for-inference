import tensorflow as tf

def load_image(input_path):
    img = tf.io.read_file(input_path)
    img = tf.image.decode_image(img)
    img = tf.image.resize(img, (224, 224))

    return img/255.0