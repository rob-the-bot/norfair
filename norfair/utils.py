import os
from typing import Sequence, Tuple

import numpy as np
import cv2
from rich import print
from rich.console import Console
from rich.table import Table
# Importing the Keras libraries and packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def validate_points(points: np.array) -> np.array:
    # If the user is tracking only a single point, reformat it slightly.
    if points.shape == (2,):
        points = points[np.newaxis, ...]
    elif len(points.shape) == 1:
        print_detection_error_message_and_exit(points)
    else:
        if points.shape[1] != 2 or len(points.shape) > 2:
            print_detection_error_message_and_exit(points)
    return points


def print_detection_error_message_and_exit(points):
    print("\n[red]INPUT ERROR:[/red]")
    print(
        f"Each `Detection` object should have a property `points` of shape (num_of_points_to_track, 2), not {points.shape}. Check your `Detection` list creation code."
    )
    print("You can read the documentation for the `Detection` class here:")
    print("https://github.com/tryolabs/norfair/tree/master/docs#detection\n")
    exit()


def print_objects_as_table(tracked_objects: Sequence):
    """Used for helping in debugging"""
    print()
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Id", style="yellow", justify="center")
    table.add_column("Age", justify="right")
    table.add_column("Hit Counter", justify="right")
    table.add_column("Last distance", justify="right")
    table.add_column("Init Id", justify="center")
    for obj in tracked_objects:
        table.add_row(
            str(obj.id),
            str(obj.age),
            str(obj.hit_counter),
            f"{obj.last_distance:.4f}",
            str(obj.initializing_id),
        )
    console.print(table)


def get_terminal_size(default: Tuple[int, int] = (80, 24)) -> Tuple[int, int]:
    columns, lines = default
    for fd in range(0, 3):  # First in order 0=Std In, 1=Std Out, 2=Std Error
        try:
            columns, lines = os.get_terminal_size(fd)
        except OSError:
            continue
        break
    return columns, lines


def get_cutout(points, image):
    """Returns a rectangular cut-out from a set of points on an image"""
    max_x = int(max(points[:, 0]))
    min_x = int(min(points[:, 0]))
    max_y = int(max(points[:, 1]))
    min_y = int(min(points[:, 1]))
    return image[min_y:max_y, min_x:max_x]


# bounding box function
def bounding_box(points):

    # remove rows having all zeroes
    data = points[~np.all(points == 0, axis=1)]
    xmin, ymin = data.min(axis=0)
    xmax, ymax = data.max(axis=0)

    return max(round(xmin), 0), max(round(ymin), 0), max(round(xmax), 0), max(round(ymax), 0)


def crop_resize(frame, points):
    xmin, ymin, xmax, ymax = bounding_box(points.round().astype(int))
    if ymin==ymax:
        ymax+=1
    if xmin==xmax:
        xmax+=1

    img = frame[ymin:ymax, xmin:xmax, :]
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, [64, 64], interpolation = cv2.INTER_AREA)
    return img#.flatten()


def build_model(IMG_SIZE: int, NUM_CLASSES: int, scratch=False):
    if scratch:
        """
        Build an EfficientNetB0 with NUM_CLASSES output classes, that is initialized from scratch
        """
        weights=None
    else:
        """
        Transfer learning from pre-trained weights
        Here we initialize the model with pre-trained ImageNet weights, and we fine-tune it on our own dataset.
        """
        weights="imagenet"
        
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    model = EfficientNetB0(include_top=False, weights=weights, input_tensor=inputs, classes=NUM_CLASSES)

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization(name="batch_norm")(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs)
    METRICS = [
        keras.metrics.TruePositives(name="tp"),
        keras.metrics.FalsePositives(name="fp"),
        keras.metrics.TrueNegatives(name="tn"),
        keras.metrics.FalseNegatives(name="fn"),
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc"),
        keras.metrics.AUC(name="prc", curve="PR"),  # precision-recall curve
    ]
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=METRICS)
    return model



def build_model_simple():
    # Initialising the CNN
    model = Sequential()
    # Step 1 - Convolution
    model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    # Step 2 - Pooling
    model.add(MaxPooling2D(pool_size = (2, 2)))
    # Adding a second convolutional layer
    model.add(Conv2D(32, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    # Step 3 - Flattening
    model.add(Flatten())
    # Step 4 - Full connection
    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dense(units = 1, activation = 'sigmoid'))
    # Compiling the CNN
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model


class DummyOpenCVImport:
    def __getattribute__(self, name):
        print(
            """[bold red]Missing dependency:[/bold red] You are trying to use Norfair's video features. However, OpenCV is not installed.

Please, make sure there is an existing installation of OpenCV or install Norfair with `pip install norfair\[video]`."""
        )
        exit()


class DummyMOTMetricsImport:
    def __getattribute__(self, name):
        print(
            """[bold red]Missing dependency:[/bold red] You are trying to use Norfair's metrics features without the required dependencies.

Please, install Norfair with `pip install norfair\[metrics]`, or `pip install norfair\[metrics,video]` if you also want video features."""
        )
        exit()
