"""
    Class file for the base model
"""

from __future__ import annotations
from typing import Any, Union

import numpy as np
import pandas as pd
import tensorflow as tf

#from nptyping import NDArray
from sklearn.base import BaseEstimator
from qkerasV3 import QActivation


class BaseModel(BaseEstimator):
    """
        BaseModel which can hold different wrapper function (_build_model, fit, ...)
        which needs to be implemented in a special model classes
    """

    def __init__(self,
            # BaseModel HPs
            hidden_layers: list[int] = [32, 16, 8],
            batch_normalisation_layers: list[int] = [32,16,8], #size and number of layers must match hidden_layers
            quantized_position: list[bool] = [False, True, False],
            batch_normalisation: bool = False,
            activation_binary: str = 'bernoulli',
            activation_nonbinary: str = 'sigmoid',
            kernel_regularizer: float = 0.0,
            drop_out: float = 0,
            # Common HPs
            batch_size: int = 200,
            learning_rate: float = 0.001,
            learning_rate_decay_rate: float = 1.,
            learning_rate_decay_steps: int = 1000,
            optimizer: str = "Adam",
            epoch: int = 10,
            loss: str = 'categorical_crossentropy',
            # other variables
            verbose: int = 0,
            validation_size: float = 0.0,
            input_shape: Union[tuple, int] = 0,
            random_seed: int = 42,
            name: str = "BaseModel",
            dataset_name: str = "Compas",
            print_summary: bool = False,
            bits: int = 2,
            checkpoint_path: str = "",
            datetime: str = "",
            run_eagerly: bool = False,
            last_layer_size: int = 1,
            conv: bool = False
        ) -> None:

        # HPs
        self.hidden_layers = hidden_layers
        self.batch_normalisation_layers = batch_normalisation_layers
        self.batch_normalisation = batch_normalisation
        self.activation_binary = activation_binary
        self.activation_nonbinary = activation_nonbinary
        self.kernel_regularizer = kernel_regularizer
        self.drop_out = drop_out
        self.batch_size = batch_size
        if loss == 'binary_crossentropy':
            self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        elif loss == 'sparse_categorical_crossentropy':
            self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        else:
            self.loss = loss
        self.learning_rate = learning_rate
        self.learning_rate_decay_rate = learning_rate_decay_rate
        self.learning_rate_decay_steps = learning_rate_decay_steps
        if optimizer == "Adam":
            self.optimizer = tf.keras.optimizers.Adam
        if optimizer == "SGD":
            self.optimizer = tf.keras.optimizers.SGD
        if optimizer == "Nadam":
            raise NotImplementedError
        self.quantized_position = quantized_position
        self.optimizer_name = optimizer
        self.epoch = epoch
        self.bits = bits
        self.checkpoint_path = checkpoint_path + "/" + datetime + "/"
        self.datetime = datetime
        self.run_eagerly = run_eagerly

        # other variables
        self.model = Any
        self.verbose = verbose
        self.validation_size = validation_size
        self.input_shape = input_shape
        self.random_seed = random_seed
        self.name = name
        self.dataset_name = dataset_name
        self.print_summary = print_summary
        self.last_layer_size = last_layer_size
        self.conv = conv

    def _plot_model(self, model: Any, output_name: str) -> None:
        """
            Plot the model and save it to png file using
            tf.keras.utils.plot_model

            Args:
                Any: model
                str: output_name

            Returns:
                None
        """

        tf.keras.utils.plot_model(
            model,
            to_file=output_name,
            show_shapes=False,
            show_dtype=False,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=96,
            layer_range=None,
            show_layer_activations=False,
        )

    def _get_quantized_activation(self, activation_str: str) -> str:
        """
            Generates the string for the binary activation using 
            self.bits for the number of bits.

            Args:
                str: activation_str

            Returns:
                str: quantized_activation
        """

        if activation_str == 'tanh':
            return f'quantized_tanh({self.bits}, symmetric=1)'
        if activation_str == 'sigmoid':
            # TODO: check for symmetric=1
            return f'quantized_sigmoid({self.bits})'
        if activation_str == 'bernoulli':
            return 'bernoulli'
        raise ValueError(f'Activation string {activation_str} not recognized')

    def _get_cls_part(
        self,
        input_layer: Any,
        num_relevance_classes: int = 2,
        feature_activation: str = "softmax",
        kernel_regularizer: Any = None,
        name: str = "cls_part",
        index: int= 0
    ) -> Any:
        """
            Build all the output layer of the model.

            Args:
                Any: input_layer
                int: num_relevance_classes
                str: feature_activation
                Any: kernel_regularizer
                str: name
                int: index

            Returns:
                Any: output_layer
        """

        out = tf.keras.layers.Dense(
            units=num_relevance_classes,
            activation=feature_activation,
            kernel_regularizer=kernel_regularizer,
            activity_regularizer=kernel_regularizer,
            name=f"{name}_{index}"
        )(input_layer)

        return out

    def _get_hidden_qlayer(
            self,
            input_layer: Any,
            hidden_layer: list[int] = [10, 5],
            drop_out: float = 0.,
            kernel_regularizer: Any = None,
            name: str = "",
            conv: bool = False,
            kernel_size: tuple = (3, 3)
    ) -> Any:
        """
            Build all the hidden layers of the model.

            Args:
                Any: input_layer
                list[int]: hidden_layer
                float: drop_out
                Any: kernel_regularizer
                str: name
                conv: True if you want a convolutional net. The filter numbers will be taken from
                hidden_layer.
                kernel_size: the filter size for convnets. Ignored if conv = False.

            Returns:
                Any: output_layer
        """

        # get the activation functions for the binary part
        activation_binary = self._get_quantized_activation(self.activation_binary)

        hidden_layers = input_layer
        last_quantized = None
        # loop over the number of hidden layers for the whole network
        for i in range(len(hidden_layer)):
            if self.batch_normalisation and i != 0:
                hidden_layers = tf.keras.layers.BatchNormalization()(hidden_layers)
            if conv:
                hidden_layers = tf.keras.layers.Conv2D(
                    filters=hidden_layer[i],
                    kernel_size=kernel_size,
                    activation=self.activation_nonbinary,
                    kernel_regularizer=kernel_regularizer,
                    activity_regularizer=kernel_regularizer,
                    name=f"{name}_{i}"
                )(hidden_layers)
            else:
                hidden_layers = tf.keras.layers.Dense(
                    units=hidden_layer[i],
                    activation=self.activation_nonbinary,
                    kernel_regularizer=kernel_regularizer,
                    activity_regularizer=kernel_regularizer,
                    name=f"{name}_{i}"
                )(hidden_layers)
            if self.quantized_position[i]:
                hidden_layers = QActivation(
                    activation_binary,
                    activity_regularizer=kernel_regularizer,
                    name=f"qact_{name}_{i}"
                )(hidden_layers)
                last_quantized = hidden_layers
            if drop_out > 0:
                hidden_layers = tf.keras.layers.Dropout(drop_out)(hidden_layers)

        # flatten if net is convolutional
        if conv:
            hidden_layers = tf.keras.layers.Flatten()(hidden_layers)

        if self.batch_normalisation:
            hidden_layers = tf.keras.layers.BatchNormalization()(hidden_layers)

        # create classification part
        output_layer = self._get_cls_part(
            input_layer=hidden_layers,
            num_relevance_classes=self.last_layer_size,
            feature_activation=self.activation_nonbinary,
            kernel_regularizer=tf.keras.regularizers.l2(self.kernel_regularizer),
            name=name,
            index=len(hidden_layer)
        )

        return output_layer, last_quantized

    def _get_hidden_layer(
            self,
            input_layer,
            hidden_layer=[10, 5],
            drop_out=0,
            feature_activation="tanh",
            last_activation="",
            reg=0,
            name="",
            build_bias=False,
            hybrid=False,
            bits=2,
            qactivation="bernoulli",
            qLayer=10,
            conv=False
    ):

        nn = input_layer
        for i in range(len(hidden_layer)):

            if not conv:
                nn = tf.keras.layers.Dense(
                    units=hidden_layer[i],
                    activation=feature_activation,
                    kernel_regularizer=tf.keras.regularizers.l2(reg),
                    bias_regularizer=tf.keras.regularizers.l2(reg),
                    activity_regularizer=tf.keras.regularizers.l2(reg),
                    name="nn_{}_{}".format(name, i)
                )(nn)
            else:
                nn = tf.keras.layers.Conv2D(
                    hidden_layer[i],
                    kernel_size=(3, 3),
                    activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(reg),
                    bias_regularizer=tf.keras.regularizers.l2(reg),
                    activity_regularizer=tf.keras.regularizers.l2(reg),
                    name="conv_{}_{}".format(name, i)
                )(nn)
                nn = tf.keras.layers.MaxPooling2D()(nn)

            if drop_out > 0 and (i < (len(hidden_layer) - 1) or last_activation != ""):
                nn = tf.keras.layers.Dropout(drop_out)(nn)

        if last_activation != "":
            nn = tf.keras.layers.Dense(
                units=1,
                activation=last_activation,
                kernel_regularizer=tf.keras.regularizers.l2(reg),
                bias_regularizer=tf.keras.regularizers.l2(reg),
                activity_regularizer=tf.keras.regularizers.l2(reg),
                name="nn_out_{}".format(name)
            )(nn)
            
        if conv:
            nn = tf.keras.layers.Flatten(name='flatten')(nn)

        if build_bias:
            hidden_part = nn
        else:
            hidden_part = tf.keras.models.Model(
                inputs=input_layer,
                outputs=nn,
                name=name
            )

        if self.verbose == 2 and not build_bias:
            hidden_part.summary()

        return hidden_part

    def _get_ranking_part(
            self,
            input_layer,
            units=1,
            feature_activation="tanh",
            reg=0,
            use_bias=False,
            name="ranking_part"
    ):

        out = tf.keras.layers.Dense(
            units=units,
            activation=feature_activation,
            use_bias=use_bias,
            kernel_regularizer=tf.keras.regularizers.l2(reg),
            activity_regularizer=tf.keras.regularizers.l2(reg),
            name=name
        )(input_layer)

        return out

    def _build_model(self) -> None:
        """
            Wrapper function which needs to be implemented in a special
            model class.

            Args:
                NoInput
            Returns:
                None
        """

        raise NotImplementedError

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, **fit_params: dict[Any, Any]) -> None:
        """
            Wrapper function which needs to be implemented in a special
            model class.

            Args:
                NDArray: x_train
                NDArray: y_train
                dict[Any]: fit_params

            Returns:
                None
        """

        raise NotImplementedError

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
            Get the class probablities.

            Args:
                NDArray: features
            Returns:
                NDArray: class probablities
        """

        if len(features.shape) == 1:
            features = [features] # type: ignore

        res = self.model.predict([features], batch_size=self.batch_size, verbose=self.verbose)[0] # type: ignore

        return res

    def predict(self, features: np.ndarray, threshold: np.ndarray) -> list[int]:
        """
            Predict a binary class value with a threshold.

            Args:
                NDArray: features
                int: threshold
            Returns:
                list[int]: predictions
        """

        if len(features.shape) == 1:
            features = [features] # type: ignore

        res = self.model.predict([features], batch_size=self.batch_size, verbose=self.verbose) # type: ignore

        return [1 if r > threshold else 0 for r in res[0]]

    def to_dict(self) -> dict[str, Any]:
        """
            Return a dictionary representation of the object while dropping the tensorflow stuff.
            Useful to keep track of hyperparameters at the experiment level.

            Args:
                None
            Returns:
                dict[str]: dict of class attributes
        """

        attr_dict = dict(vars(self))

        for key in ['optimizer']:
            try:
                attr_dict.pop(key)
            except KeyError:
                pass

        return attr_dict

    def get_complexity(self) -> int:
        """
            Returns the number of trainable weights of the model.

            Args:
                None
            Returns:
                int: number of trainable weights
        """

        return int(np.sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])) # type: ignore

    def convert_array_to_float(self, array):
        if isinstance(array, np.ndarray):
            array = array.astype('float32')
        if isinstance(array, pd.DataFrame):
            array = array.values.astype('float32')
        return array
