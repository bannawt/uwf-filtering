from keras.applications import InceptionResNetV2
from keras.models import Model
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

idg = ImageDataGenerator(
    rescale=1./255,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=(0.8, 1.6),
)
tdg = ImageDataGenerator(rescale=1./255)

base = InceptionResNetV2(input_shape=(512, 512, 3), include_top=False, weights="imagenet", pooling="avg")
x = base.output
x = Dense(1024, activation="relu")(x)
x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base.input, outputs=x)
model.compile("adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit_generator(
    idg.flow_from_directory("data/train",
        class_mode="binary",
        target_size=(512, 512),
        batch_size=10,
    ),
    steps_per_epoch=12621,
    epochs=180,
    validation_data=tdg.flow_from_directory("data/val",
        class_mode="binary",
        target_size=(512, 512),
        batch_size=10,
    ),
    validation_steps=542,
    callbacks=[EarlyStopping(patience=60, restore_best_weights=True)],
)
model.save("model.hdf5")
