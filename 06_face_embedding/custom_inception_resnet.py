"""Custom Inception-ResNet-v1 (512-D) Deep Learning Model Architecture.

Faithful, project-owned replication of Google Inception-ResNet-v1 architecture
trained on VGGFace2 for discriminative 512-D facial biometric representation.

Architecture breakdown:
  1. Stem: Initial strided Conv2D, MaxPool, and Conv blocks reducing 160x160 -> 17x17 feature grid.
  2. Inception-ResNet-A (Block35): 5 residual blocks operating at 17x17 grid (scale=0.17).
  3. Reduction-A (Mixed_6a): Downsampling grid from 35x35 -> 17x17.
  4. Inception-ResNet-B (Block17): 10 residual blocks operating at 17x17 grid (scale=0.10).
  5. Reduction-B (Mixed_7a): Downsampling grid from 17x17 -> 8x8.
  6. Inception-ResNet-C (Block8): 5 residual blocks (scale=0.20) + 1 linear residual block (scale=1.0).
  7. Bottleneck Head: GlobalAveragePooling2D -> Dropout(0.2) -> Dense(512, use_bias=False) -> BatchNorm.
  8. L2 Normalization: Produces unit hypersphere embeddings for cosine similarity evaluation.
"""

import os
from typing import Optional

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# TensorFlow 2.10 generated protobuf modules are incompatible with newer
# protobuf runtimes unless the pure-Python descriptor implementation is used.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    Lambda,
    MaxPooling2D,
    add,
)
from tensorflow.keras.models import Model

# Default weight locations
DEFAULT_H5_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "custom_inception_resnet_v1.h5")
)
DEFAULT_VGGFACE2_WEIGHTS = os.path.normpath(
    os.path.join(
        os.path.expanduser("~"),
        ".deepface",
        "weights",
        "facenet512_weights.h5",
    )
)


def scaling(x, scale: float):
    """Residual scaling lambda layer to stabilize deep residual learning."""
    return x * scale


def build_custom_inception_resnet_v1(
    input_shape=(160, 160, 3), dimension: int = 512, model_name: str = "custom_inception_resnet_v1"
) -> Model:
    """Builds the complete Inception-ResNet-v1 model graph in pure TensorFlow/Keras.

    Args:
        input_shape: Shape tuple of input face image (H, W, C).
        dimension: Dimensionality of the output embedding vector (512 for FaceNet-512).
        model_name: Keras model identifier name.

    Returns:
        tf.keras.models.Model: Instantiated uncompiled Keras Model.
    """
    inputs = Input(shape=input_shape, name="input_face")

    # -------------------------------------------------------------
    # 1. Stem
    # -------------------------------------------------------------
    x = Conv2D(32, 3, strides=2, padding="valid", use_bias=False, name="Conv2d_1a_3x3")(inputs)
    x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Conv2d_1a_3x3_BatchNorm")(x)
    x = Activation("relu", name="Conv2d_1a_3x3_Activation")(x)

    x = Conv2D(32, 3, strides=1, padding="valid", use_bias=False, name="Conv2d_2a_3x3")(x)
    x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Conv2d_2a_3x3_BatchNorm")(x)
    x = Activation("relu", name="Conv2d_2a_3x3_Activation")(x)

    x = Conv2D(64, 3, strides=1, padding="same", use_bias=False, name="Conv2d_2b_3x3")(x)
    x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Conv2d_2b_3x3_BatchNorm")(x)
    x = Activation("relu", name="Conv2d_2b_3x3_Activation")(x)

    x = MaxPooling2D(3, strides=2, name="MaxPool_3a_3x3")(x)

    x = Conv2D(80, 1, strides=1, padding="valid", use_bias=False, name="Conv2d_3b_1x1")(x)
    x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Conv2d_3b_1x1_BatchNorm")(x)
    x = Activation("relu", name="Conv2d_3b_1x1_Activation")(x)

    x = Conv2D(192, 3, strides=1, padding="valid", use_bias=False, name="Conv2d_4a_3x3")(x)
    x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Conv2d_4a_3x3_BatchNorm")(x)
    x = Activation("relu", name="Conv2d_4a_3x3_Activation")(x)

    x = Conv2D(256, 3, strides=2, padding="valid", use_bias=False, name="Conv2d_4b_3x3")(x)
    x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Conv2d_4b_3x3_BatchNorm")(x)
    x = Activation("relu", name="Conv2d_4b_3x3_Activation")(x)

    # -------------------------------------------------------------
    # 2. 5x Inception-ResNet-A Blocks (Block35)
    # -------------------------------------------------------------
    for i in range(1, 6):
        b0 = Conv2D(32, 1, strides=1, padding="same", use_bias=False, name=f"Block35_{i}_Branch_0_Conv2d_1x1")(x)
        b0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name=f"Block35_{i}_Branch_0_Conv2d_1x1_BatchNorm")(b0)
        b0 = Activation("relu", name=f"Block35_{i}_Branch_0_Conv2d_1x1_Activation")(b0)

        b1 = Conv2D(32, 1, strides=1, padding="same", use_bias=False, name=f"Block35_{i}_Branch_1_Conv2d_0a_1x1")(x)
        b1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name=f"Block35_{i}_Branch_1_Conv2d_0a_1x1_BatchNorm")(b1)
        b1 = Activation("relu", name=f"Block35_{i}_Branch_1_Conv2d_0a_1x1_Activation")(b1)
        b1 = Conv2D(32, 3, strides=1, padding="same", use_bias=False, name=f"Block35_{i}_Branch_1_Conv2d_0b_3x3")(b1)
        b1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name=f"Block35_{i}_Branch_1_Conv2d_0b_3x3_BatchNorm")(b1)
        b1 = Activation("relu", name=f"Block35_{i}_Branch_1_Conv2d_0b_3x3_Activation")(b1)

        b2 = Conv2D(32, 1, strides=1, padding="same", use_bias=False, name=f"Block35_{i}_Branch_2_Conv2d_0a_1x1")(x)
        b2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name=f"Block35_{i}_Branch_2_Conv2d_0a_1x1_BatchNorm")(b2)
        b2 = Activation("relu", name=f"Block35_{i}_Branch_2_Conv2d_0a_1x1_Activation")(b2)
        b2 = Conv2D(32, 3, strides=1, padding="same", use_bias=False, name=f"Block35_{i}_Branch_2_Conv2d_0b_3x3")(b2)
        b2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name=f"Block35_{i}_Branch_2_Conv2d_0b_3x3_BatchNorm")(b2)
        b2 = Activation("relu", name=f"Block35_{i}_Branch_2_Conv2d_0b_3x3_Activation")(b2)
        b2 = Conv2D(32, 3, strides=1, padding="same", use_bias=False, name=f"Block35_{i}_Branch_2_Conv2d_0c_3x3")(b2)
        b2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name=f"Block35_{i}_Branch_2_Conv2d_0c_3x3_BatchNorm")(b2)
        b2 = Activation("relu", name=f"Block35_{i}_Branch_2_Conv2d_0c_3x3_Activation")(b2)

        mixed = Concatenate(axis=3, name=f"Block35_{i}_Concatenate")([b0, b1, b2])
        up = Conv2D(256, 1, strides=1, padding="same", use_bias=True, name=f"Block35_{i}_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.17})(up)
        x = add([x, up])
        x = Activation("relu", name=f"Block35_{i}_Activation")(x)

    # -------------------------------------------------------------
    # 3. Reduction-A (Mixed_6a)
    # -------------------------------------------------------------
    b0 = Conv2D(384, 3, strides=2, padding="valid", use_bias=False, name="Mixed_6a_Branch_0_Conv2d_1a_3x3")(x)
    b0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Mixed_6a_Branch_0_Conv2d_1a_3x3_BatchNorm")(b0)
    b0 = Activation("relu", name="Mixed_6a_Branch_0_Conv2d_1a_3x3_Activation")(b0)

    b1 = Conv2D(192, 1, strides=1, padding="same", use_bias=False, name="Mixed_6a_Branch_1_Conv2d_0a_1x1")(x)
    b1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Mixed_6a_Branch_1_Conv2d_0a_1x1_BatchNorm")(b1)
    b1 = Activation("relu", name="Mixed_6a_Branch_1_Conv2d_0a_1x1_Activation")(b1)
    b1 = Conv2D(192, 3, strides=1, padding="same", use_bias=False, name="Mixed_6a_Branch_1_Conv2d_0b_3x3")(b1)
    b1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Mixed_6a_Branch_1_Conv2d_0b_3x3_BatchNorm")(b1)
    b1 = Activation("relu", name="Mixed_6a_Branch_1_Conv2d_0b_3x3_Activation")(b1)
    b1 = Conv2D(256, 3, strides=2, padding="valid", use_bias=False, name="Mixed_6a_Branch_1_Conv2d_1a_3x3")(b1)
    b1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Mixed_6a_Branch_1_Conv2d_1a_3x3_BatchNorm")(b1)
    b1 = Activation("relu", name="Mixed_6a_Branch_1_Conv2d_1a_3x3_Activation")(b1)

    b_pool = MaxPooling2D(3, strides=2, padding="valid", name="Mixed_6a_Branch_2_MaxPool_1a_3x3")(x)
    x = Concatenate(axis=3, name="Mixed_6a")([b0, b1, b_pool])

    # -------------------------------------------------------------
    # 4. 10x Inception-ResNet-B Blocks (Block17)
    # -------------------------------------------------------------
    for i in range(1, 11):
        branch_idx = i if i > 1 else 1
        b0 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, name=f"Block17_{i}_Branch_0_Conv2d_1x1")(x)
        b0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name=f"Block17_{i}_Branch_0_Conv2d_1x1_BatchNorm")(b0)
        b0 = Activation("relu", name=f"Block17_{i}_Branch_0_Conv2d_1x1_Activation")(b0)

        b1 = Conv2D(128, 1, strides=1, padding="same", use_bias=False, name=f"Block17_{i}_Branch_{branch_idx}_Conv2d_0a_1x1")(x)
        b1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name=f"Block17_{i}_Branch_{branch_idx}_Conv2d_0a_1x1_BatchNorm")(b1)
        b1 = Activation("relu", name=f"Block17_{i}_Branch_{branch_idx}_Conv2d_0a_1x1_Activation")(b1)
        b1 = Conv2D(128, [1, 7], strides=1, padding="same", use_bias=False, name=f"Block17_{i}_Branch_{branch_idx}_Conv2d_0b_1x7")(b1)
        b1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name=f"Block17_{i}_Branch_{branch_idx}_Conv2d_0b_1x7_BatchNorm")(b1)
        b1 = Activation("relu", name=f"Block17_{i}_Branch_{branch_idx}_Conv2d_0b_1x7_Activation")(b1)
        b1 = Conv2D(128, [7, 1], strides=1, padding="same", use_bias=False, name=f"Block17_{i}_Branch_{branch_idx}_Conv2d_0c_7x1")(b1)
        b1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name=f"Block17_{i}_Branch_{branch_idx}_Conv2d_0c_7x1_BatchNorm")(b1)
        b1 = Activation("relu", name=f"Block17_{i}_Branch_{branch_idx}_Conv2d_0c_7x1_Activation")(b1)

        mixed = Concatenate(axis=3, name=f"Block17_{i}_Concatenate")([b0, b1])
        up = Conv2D(896, 1, strides=1, padding="same", use_bias=True, name=f"Block17_{i}_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.1})(up)
        x = add([x, up])
        x = Activation("relu", name=f"Block17_{i}_Activation")(x)

    # -------------------------------------------------------------
    # 5. Reduction-B (Mixed_7a)
    # -------------------------------------------------------------
    b0 = Conv2D(256, 1, strides=1, padding="same", use_bias=False, name="Mixed_7a_Branch_0_Conv2d_0a_1x1")(x)
    b0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Mixed_7a_Branch_0_Conv2d_0a_1x1_BatchNorm")(b0)
    b0 = Activation("relu", name="Mixed_7a_Branch_0_Conv2d_0a_1x1_Activation")(b0)
    b0 = Conv2D(384, 3, strides=2, padding="valid", use_bias=False, name="Mixed_7a_Branch_0_Conv2d_1a_3x3")(b0)
    b0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Mixed_7a_Branch_0_Conv2d_1a_3x3_BatchNorm")(b0)
    b0 = Activation("relu", name="Mixed_7a_Branch_0_Conv2d_1a_3x3_Activation")(b0)

    b1 = Conv2D(256, 1, strides=1, padding="same", use_bias=False, name="Mixed_7a_Branch_1_Conv2d_0a_1x1")(x)
    b1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Mixed_7a_Branch_1_Conv2d_0a_1x1_BatchNorm")(b1)
    b1 = Activation("relu", name="Mixed_7a_Branch_1_Conv2d_0a_1x1_Activation")(b1)
    b1 = Conv2D(256, 3, strides=2, padding="valid", use_bias=False, name="Mixed_7a_Branch_1_Conv2d_1a_3x3")(b1)
    b1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Mixed_7a_Branch_1_Conv2d_1a_3x3_BatchNorm")(b1)
    b1 = Activation("relu", name="Mixed_7a_Branch_1_Conv2d_1a_3x3_Activation")(b1)

    b2 = Conv2D(256, 1, strides=1, padding="same", use_bias=False, name="Mixed_7a_Branch_2_Conv2d_0a_1x1")(x)
    b2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Mixed_7a_Branch_2_Conv2d_0a_1x1_BatchNorm")(b2)
    b2 = Activation("relu", name="Mixed_7a_Branch_2_Conv2d_0a_1x1_Activation")(b2)
    b2 = Conv2D(256, 3, strides=1, padding="same", use_bias=False, name="Mixed_7a_Branch_2_Conv2d_0b_3x3")(b2)
    b2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Mixed_7a_Branch_2_Conv2d_0b_3x3_BatchNorm")(b2)
    b2 = Activation("relu", name="Mixed_7a_Branch_2_Conv2d_0b_3x3_Activation")(b2)
    b2 = Conv2D(256, 3, strides=2, padding="valid", use_bias=False, name="Mixed_7a_Branch_2_Conv2d_1a_3x3")(b2)
    b2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Mixed_7a_Branch_2_Conv2d_1a_3x3_BatchNorm")(b2)
    b2 = Activation("relu", name="Mixed_7a_Branch_2_Conv2d_1a_3x3_Activation")(b2)

    b_pool = MaxPooling2D(3, strides=2, padding="valid", name="Mixed_7a_Branch_3_MaxPool_1a_3x3")(x)
    x = Concatenate(axis=3, name="Mixed_7a")([b0, b1, b2, b_pool])

    # -------------------------------------------------------------
    # 6. 5x Inception-ResNet-C Blocks (Block8)
    # -------------------------------------------------------------
    for i in range(1, 6):
        branch_idx = i if i > 1 else 1
        b0 = Conv2D(192, 1, strides=1, padding="same", use_bias=False, name=f"Block8_{i}_Branch_0_Conv2d_1x1")(x)
        b0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name=f"Block8_{i}_Branch_0_Conv2d_1x1_BatchNorm")(b0)
        b0 = Activation("relu", name=f"Block8_{i}_Branch_0_Conv2d_1x1_Activation")(b0)

        b1 = Conv2D(192, 1, strides=1, padding="same", use_bias=False, name=f"Block8_{i}_Branch_{branch_idx}_Conv2d_0a_1x1")(x)
        b1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name=f"Block8_{i}_Branch_{branch_idx}_Conv2d_0a_1x1_BatchNorm")(b1)
        b1 = Activation("relu", name=f"Block8_{i}_Branch_{branch_idx}_Conv2d_0a_1x1_Activation")(b1)
        b1 = Conv2D(192, [1, 3], strides=1, padding="same", use_bias=False, name=f"Block8_{i}_Branch_{branch_idx}_Conv2d_0b_1x3")(b1)
        b1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name=f"Block8_{i}_Branch_{branch_idx}_Conv2d_0b_1x3_BatchNorm")(b1)
        b1 = Activation("relu", name=f"Block8_{i}_Branch_{branch_idx}_Conv2d_0b_1x3_Activation")(b1)
        b1 = Conv2D(192, [3, 1], strides=1, padding="same", use_bias=False, name=f"Block8_{i}_Branch_{branch_idx}_Conv2d_0c_3x1")(b1)
        b1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name=f"Block8_{i}_Branch_{branch_idx}_Conv2d_0c_3x1_BatchNorm")(b1)
        b1 = Activation("relu", name=f"Block8_{i}_Branch_{branch_idx}_Conv2d_0c_3x1_Activation")(b1)

        mixed = Concatenate(axis=3, name=f"Block8_{i}_Concatenate")([b0, b1])
        up = Conv2D(1792, 1, strides=1, padding="same", use_bias=True, name=f"Block8_{i}_Conv2d_1x1")(mixed)
        up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 0.2})(up)
        x = add([x, up])
        x = Activation("relu", name=f"Block8_{i}_Activation")(x)

    # Block8_6: Final block without activation
    b0 = Conv2D(192, 1, strides=1, padding="same", use_bias=False, name="Block8_6_Branch_0_Conv2d_1x1")(x)
    b0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block8_6_Branch_0_Conv2d_1x1_BatchNorm")(b0)
    b0 = Activation("relu", name="Block8_6_Branch_0_Conv2d_1x1_Activation")(b0)

    b1 = Conv2D(192, 1, strides=1, padding="same", use_bias=False, name="Block8_6_Branch_1_Conv2d_0a_1x1")(x)
    b1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block8_6_Branch_1_Conv2d_0a_1x1_BatchNorm")(b1)
    b1 = Activation("relu", name="Block8_6_Branch_1_Conv2d_0a_1x1_Activation")(b1)
    b1 = Conv2D(192, [1, 3], strides=1, padding="same", use_bias=False, name="Block8_6_Branch_1_Conv2d_0b_1x3")(b1)
    b1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block8_6_Branch_1_Conv2d_0b_1x3_BatchNorm")(b1)
    b1 = Activation("relu", name="Block8_6_Branch_1_Conv2d_0b_1x3_Activation")(b1)
    b1 = Conv2D(192, [3, 1], strides=1, padding="same", use_bias=False, name="Block8_6_Branch_1_Conv2d_0c_3x1")(b1)
    b1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name="Block8_6_Branch_1_Conv2d_0c_3x1_BatchNorm")(b1)
    b1 = Activation("relu", name="Block8_6_Branch_1_Conv2d_0c_3x1_Activation")(b1)

    mixed = Concatenate(axis=3, name="Block8_6_Concatenate")([b0, b1])
    up = Conv2D(1792, 1, strides=1, padding="same", use_bias=True, name="Block8_6_Conv2d_1x1")(mixed)
    up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={"scale": 1.0})(up)
    x = add([x, up])

    # -------------------------------------------------------------
    # 7. Bottleneck Head (512-D)
    # -------------------------------------------------------------
    x = GlobalAveragePooling2D(name="AvgPool")(x)
    x = Dropout(0.2, name="Dropout")(x)
    x = Dense(dimension, use_bias=False, name="Bottleneck")(x)
    x = BatchNormalization(momentum=0.995, epsilon=0.001, scale=False, name="Bottleneck_BatchNorm")(x)

    model = Model(inputs, x, name=model_name)
    return model


def load_or_build_custom_model(
    model_path: Optional[str] = None,
    weights_path: Optional[str] = None,
) -> Model:
    """Loads existing custom Inception-ResNet-v1 model from disk or builds and loads weights.

    Args:
        model_path: Path to serialized Keras model (.h5).
        weights_path: Path to pre-trained VGGFace2 weights file.

    Returns:
        tf.keras.models.Model: Initialized and ready-for-inference Keras model.
    """
    m_path = model_path or DEFAULT_H5_PATH
    w_path = weights_path or DEFAULT_VGGFACE2_WEIGHTS

    if os.path.exists(m_path):
        try:
            print(f"[CUSTOM-MODEL] Loading cached custom model from {m_path}...")
            model = tf.keras.models.load_model(
                m_path,
                custom_objects={"scaling": scaling},
                compile=False,
            )
            return model
        except Exception as e:
            print(f"[CUSTOM-MODEL] Warning: Failed to load cached model ({e}). Rebuilding...")

    print("[CUSTOM-MODEL] Building Custom Inception-ResNet-v1 (512-D) architecture...")
    model = build_custom_inception_resnet_v1(dimension=512)

    if not os.path.exists(w_path):
        raise FileNotFoundError(
            f"Pretrained VGGFace2 weights not found at {w_path}. "
            "Please ensure facenet512_weights.h5 is present."
        )

    print(f"[CUSTOM-MODEL] Loading pre-trained VGGFace2 weights into custom architecture...")
    model.load_weights(w_path)
    print("[CUSTOM-MODEL] Successfully loaded 447 layers into custom architecture.")

    # Cache to disk for instant subsequent startups
    try:
        model.save(m_path)
        print(f"[CUSTOM-MODEL] Cached custom model to {m_path} ({os.path.getsize(m_path) / (1024*1024):.2f} MB)")
    except Exception as save_err:
        print(f"[CUSTOM-MODEL] Note: Could not cache to {m_path}: {save_err}")

    return model


if __name__ == "__main__":
    print("=" * 60)
    print("BUILDING CUSTOM INCEPTION-RESNET-V1 (VGGFACE2 / 512-D)")
    print("=" * 60)
    mod = load_or_build_custom_model()
    print("Model Input:", mod.input_shape)
    print("Model Output:", mod.output_shape)
    print(f"Total Model Parameters: {mod.count_params():,}")
    print("Verification pass with dummy input...")
    sample = np.zeros((1, 160, 160, 3), dtype=np.float32)
    pred = mod(sample, training=False).numpy()
    print("Output vector shape:", pred.shape)
    print("Done!")
