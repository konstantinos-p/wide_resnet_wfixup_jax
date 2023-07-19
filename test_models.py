from unittest import TestCase
from models import FixupBias, FixupMultiplier, FixupBasicBlock, FixupNetworkBlock, FixupWideResNet
import jax
import jax.numpy as jnp
from jax import random
from typing import Any


ModuleDef = Any
KeyArray = random.KeyArray
Array = Any
DTypeLikeFloat = Any
DTypeLikeComplex = Any
DTypeLikeInexact = Any
RealNumeric = Any


class TestFixupBias(TestCase):
    """
    Tests the FixupBias module.
    """
    def setUp(self) -> None:
        self.rng = jax.random.PRNGKey(0)
        self.constant_init = 2
        self.constant_input = 1
        self.model = FixupBias(bias_init=jax.nn.initializers.constant(self.constant_init))
        self.x = self.constant_input*jnp.ones([1, 32, 32, 3])
        self.params = self.model.init(self.rng, self.x)
        self.y = self.model.apply(self.params, self.x)


class TestFixupBiasProperties(TestFixupBias):
    def test_shape(self):
        """
        Test that the input and output shapes are equal.
        Returns
        -------

        """
        self.assertEqual(self.y.shape, self.x.shape)

    def test_values(self):
        """
        Test that the bias has been added to all elements of the array.
        Returns
        -------

        """
        self.assertEqual(jnp.all(self.constant_init+self.constant_input == self.y), True)


class TestFixupMultiplier(TestCase):
    """
    Tests the FixupMultiplier module.
    """
    def setUp(self) -> None:
        self.rng = jax.random.PRNGKey(0)
        self.constant_init = 2
        self.constant_input = 1
        self.model = FixupMultiplier(multiplier_init=jax.nn.initializers.constant(self.constant_init))
        self.x = self.constant_input*jnp.ones([1, 32, 32, 3])
        self.params = self.model.init(self.rng, self.x)
        self.y = self.model.apply(self.params, self.x)


class TestFixupMultiplierProperties(TestFixupMultiplier):
    def test_shape(self):
        """
        Test that the input and output shapes are equal.
        Returns
        -------

        """
        self.assertEqual(self.y.shape, self.x.shape)

    def test_values(self):
        """
        Test that the multiplier has been applied to all elements of the array.
        Returns
        -------

        """
        self.assertEqual(jnp.all(self.constant_init == self.y), True)


class TestFixupBasicBlock(TestCase):
    """
    Tests the FixupWideResNet module.
    """
    def setUp(self) -> None:
        self.rng = jax.random.PRNGKey(0)
        input_channels = 3
        self.model = FixupBasicBlock(in_planes=input_channels, out_planes=3, stride=1, dropRate=0.1, total_num_basic_blocks=1)
        self.x = jnp.ones([1, 32, 32, input_channels])
        self.params = self.model.init(self.rng, self.x, train=False)
        self.y = self.model.apply(self.params, self.x, train=False)


class TestFixupBasicBlockProperties(TestFixupBasicBlock):
    def test_nan(self):
        """
        Test whether the outputs of the basic block are NaN.
        Returns
        -------

        """
        self.assertEqual(jnp.any(jnp.isnan(self.y)), False)

    def test_zeros(self):
        """
        Test whether the outputs of the basic block are all zero.
        Returns
        -------

        """
        self.assertEqual(jnp.all(0 == self.y), False)


class TestFixupBlock(TestCase):
    """
    Tests the FixupWideResNet module.
    """
    def setUp(self) -> None:
        self.rng = jax.random.PRNGKey(0)
        input_channels = 3
        self.model = FixupNetworkBlock(nb_layers=1, in_planes=input_channels, out_planes=3, block_cls=FixupBasicBlock, stride=1,
                                       dropRate=0.1, total_num_basic_blocks=1)
        self.x = jnp.ones([1, 32, 32, input_channels])
        self.params = self.model.init(self.rng, self.x, train=False)
        self.y = self.model.apply(self.params, self.x, train=False)


class TestFixupBlockProperties(TestFixupBlock):
    def test_nan(self):
        """
        Test whether the outputs of the block are NaN.
        Returns
        -------

        """
        self.assertEqual(jnp.any(jnp.isnan(self.y)), False)

    def test_zeros(self):
        """
        Test whether the outputs of the block are all zero.
        Returns
        -------

        """
        self.assertEqual(jnp.all(0 == self.y), False)


class TestFixupWideResNet(TestCase):
    """
    Tests the FixupWideResNet module.
    """
    def setUp(self) -> None:
        self.rng = jax.random.PRNGKey(0)
        input_channels = 3
        self.model = FixupWideResNet(depth=10, widen_factor=1, num_classes=10, dropRate=0.3)
        self.x = jnp.ones([1, 32, 32, input_channels])
        self.params = self.model.init(self.rng, self.x, train=False)
        self.y = self.model.apply(self.params, self.x, train=False)


class TestFixupWideResNetProperties(TestFixupWideResNet):
    def test_nan(self):
        """
        Test whether the outputs of the neural network are NaN.
        Returns
        -------

        """
        self.assertEqual(jnp.any(jnp.isnan(self.y)), False)

    def test_zeros(self):
        """
        Test whether the outputs of the neural network are all zero.
        Returns
        -------

        """
        self.assertEqual(jnp.all(0 == self.y), True)


