import matplotlib.pyplot as plt


title = "Reconstructed Images and Digit Positions"

row_labels = [
    "Ground Truth",
    "Standard",
    "CoordConv"
]

scale = 3


def _generate_figure_from_rows(rows):
    num_samples = len(rows[0][0])

    fig = plt.figure(figsize=(num_samples * scale, len(rows) * scale))
    fig.suptitle(title, fontsize=10 * scale)
    plt.gray()

    for y, (images, positions) in enumerate(rows):
        for x, (image, pos) in enumerate(zip(images, positions)):
            axes = plt.subplot(len(rows), num_samples, y * num_samples + x + 1)

            plt.imshow(image)
            plt.scatter([pos[0]], [pos[1]], marker="x", c="r")

            axes.get_xaxis().set_visible(False)
            plt.yticks([])
            if x == 0:
                plt.ylabel(row_labels[y], fontsize=7 * scale)

    plt.savefig("figure.png")
    plt.show()


def generate_figure(images, positions, model_sets):
    rows = [(images, positions)]
    for encoder, decoder, probe in model_sets:
        embeddings = encoder.predict(images)
        reconstructions = decoder.predict(embeddings)
        predicted_positions = probe.predict(embeddings)

        rows.append((reconstructions, predicted_positions))

    _generate_figure_from_rows(rows)
