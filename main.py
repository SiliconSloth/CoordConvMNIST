from keras import layers
from keras.models import Model
from keras.optimizers import Adam

from tqdm.keras import TqdmCallback

from coord import CoordinateChannel2D
from dataset import generate_dataset
from figure import generate_figure


image_shape = 64, 64
embedding_size = 50


def make_encoder(use_coordconv):
    input = layers.Input(shape=(image_shape[0], image_shape[1], 1))
    x = input

    if use_coordconv:
        x = CoordinateChannel2D()(x)
    x = layers.Conv2D(32, (3, 3), activation="elu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(embedding_size, (3, 3), activation="elu", padding="same")(x)

    if use_coordconv:
        # We do not use a dense layer as it would learn position-specific weights.
        # In contrast, global pooling treats all parts of the image the same way.
        x = layers.GlobalMaxPool2D()(x)
    else:
        x = layers.Flatten()(x)
        x = layers.Dense(embedding_size)(x)

    return Model(input, x)


def make_decoder(use_coordconv):
    input = layers.Input(shape=embedding_size)
    x = input

    start_shape = image_shape[0] // 2, image_shape[1] // 2
    if use_coordconv:
        # The same input data is supplied to all parts of the
        # initial image, regardless of position. This encourages
        # the convolution layers to make use of the CoordConv
        # position channels.
        x = layers.RepeatVector(start_shape[0]*start_shape[1])(x)
    else:
        x = layers.Dense(start_shape[0] * start_shape[1] * embedding_size, activation="elu")(x)
    x = layers.Reshape((start_shape[0], start_shape[1], embedding_size))(x)

    if use_coordconv:
        x = CoordinateChannel2D()(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="elu", padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=1, activation="elu", padding="same")(x)
    x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

    return Model(input, x)


def make_probe():
    # Linear model from embeddings to positions.
    input = layers.Input(shape=embedding_size)
    output = layers.Dense(2)(input)
    return Model(input, output)


def train_autoencoder(images, use_coordconv):
    encoder = make_encoder(use_coordconv)
    decoder = make_decoder(use_coordconv)

    input = layers.Input(shape=(image_shape[0], image_shape[1], 1))
    embedding = encoder(input)
    output = decoder(embedding)

    autoencoder = Model(input, output)
    autoencoder.compile(optimizer=Adam(beta_1=0.7), loss="binary_crossentropy")

    autoencoder.fit(callbacks=[TqdmCallback(verbose=0)],
        x=images,
        y=images,
        epochs=100,
        batch_size=128,
        shuffle=True,
        verbose=0,
    )

    return encoder, decoder


def train_probe(embeddings, positions):
    probe = make_probe()

    probe.compile(optimizer="adam", loss="mean_squared_error")
    probe.fit(callbacks=[TqdmCallback(verbose=0)],
        x=embeddings,
        y=positions,
        epochs=50,
        batch_size=128,
        shuffle=True,
        verbose=0,
    )

    return probe


def train_models(model_path, images, positions, use_coordconv):
    encoder, decoder = train_autoencoder(images, use_coordconv)

    encoder.save(model_path + "/encoder")
    decoder.save(model_path + "/decoder")

    print("Generating embeddings...")
    embeddings = encoder.predict(images)

    print("Training probe...")
    probe = train_probe(embeddings, positions)
    probe.save(model_path + "/probe")

    return encoder, decoder, probe


if __name__ == "__main__":
    print("Generating dataset...")
    (train_images, train_positions), (test_images, test_positions) = generate_dataset(image_shape)

    print("Training standard model...")
    standard_models  = train_models("standard", train_images, train_positions, use_coordconv=False)

    print("Training CoordConv model...")
    coordconv_models = train_models("coordconv", train_images, train_positions, use_coordconv=True)

    print("Generating figure...")
    generate_figure(test_images[:6], test_positions[:6], [standard_models, coordconv_models])
