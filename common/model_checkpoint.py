import pickle
import tensorflow as tf


class MyModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        # Also save the optimizer state
        filepath = self._get_file_path(epoch=epoch, logs=logs, batch=None)
        filepath = filepath.rsplit(".", 1)[0]
        filepath += ".pkl"
        with open(filepath, "wb") as fp:
            pickle.dump(
                {
                    "opt": model.optimizer.get_config(),
                    "epoch": epoch + 1
                    # Add additional keys if you need to store more values
                },
                fp,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print("\nEpoch %05d: saving optimizer to %s" % (epoch + 1, filepath))


def load_model_data(model_path, opt_path):
    model = tf.keras.models.load_model(model_path)
    with open(opt_path, "rb") as fp:
        d = pickle.load(fp)
        epoch = d["epoch"]
        opt = d["opt"]
        return epoch, model, opt


# model_path = '/content/saved_model/model-{epoch:02d}-{val_accuracy:.4f}.hd5'

# checkpoint = MyModelCheckpoint(
#     filepath=model_path,
#     monitor='val_accuracy',
#     mode='max',
#     save_best_only=True,
#     verbose=1
# )
