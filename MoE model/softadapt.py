"""Implementaion of the base class for SoftAdapt."""

import torch
from typing import Tuple
import numpy
from findiff import coefficients

_EPSILON = 1e-08

"""Definition of constants for odd finite difference (up to 5)"""

# All constants are for forward finite difference method.
_FIRST_ORDER_COEFFICIENTS = numpy.array((-1, 1))
_THIRD_ORDER_COEFFICIENTS = numpy.array((-11/6, 3, -3/2, 1/3))
_FIFTH_ORDER_COEFFICIENTS = numpy.array((-137/60, 5, -5, 10/3, -5/4, 1/5))

def _get_finite_difference(input_array: numpy.array,
                           order: int = None,
                           verbose: bool = True):
    """Internal utility method for estimating rate of change.

    This function aims to approximate the rate of change for a loss function,
    which is used for the 'LossWeighted' and 'Normalized' variants of SoftAdapt.

    For even accuracy orders, we take advantage of the `findiff` package
    (https://findiff.readthedocs.io/en/latest/source/examples-basic.html).
    Accuracy orders of 1 (trivial), 3, and 5 are retrieved from an internal
    constants file. Due to the underlying mathematics of computing the
    coefficients, all accuracy orders higher than 5 must be an even number.

    Args:
        input_array: An array of floats containing loss evaluations at the
          previous 'n' points (as many points as the order) of the finite
          difference method.
        order: An integer indicating the order of the finite difference method
          we want to use. The function will use the length of the 'input_array'
          array if no values is provided.
        verbose: Whether we want the function to print out information about
          computations or not.

    Returns:
        A float which is the approximated rate of change between the loss
        points.

    Raises:
        ValueError: If the number of points in the `input_array` array is
          smaller than the order of accuracy we desire.
        Value Error: If the order of accuracy is higher than 5 and it is not an
          even number.
    """
    # First, we want to check the order and the number of loss points we are
    # given
    if order is None:
        order = len(input_array) - 1
        if verbose:
            print(f"==> Interpreting finite difference order as {order} since"
                  "no explicit order was specified.")
    else:
        if order > len(input_array):
            raise ValueError("The order of finite difference computations can"
                             "not be larger than the number of loss points. "
                             "Please check the order argument or wait until "
                             "enough points have been stored before calling the"
                             " method.")
        elif order + 1 < len(input_array):
            # print(f"==> There are more points than 'order' + 1 ({order + 1}) "
            #       f"points (array contains {len(input_array)} values). Function"
            #       f"will use the last {order} elements of loss points for "
            #       "computations.")
            input_array = input_array[(-1*order - 1):]

    order_is_even = order % 2 == 0
    # Next, we want to retrieve the correct coefficients based on the order
    if order > 5 and not order_is_even:
        raise ValueError("Accuracy orders larger than 5 must be even. Please "
                         "check the arguments passed to the function.")

    if order_is_even:
        constants = coefficients(deriv=1, acc=order)["forward"]["coefficients"]

    else:
        if order == 1:
            constants = _FIRST_ORDER_COEFFICIENTS
        elif order == 3:
            constants = _THIRD_ORDER_COEFFICIENTS
        else:
            constants = _FIFTH_ORDER_COEFFICIENTS

    pointwise_multiplication = [
        input_array[i] * constants[i] for i in range(len(constants))
    ]
    return numpy.sum(pointwise_multiplication)

class SoftAdaptBase():
    """Base model for any of the SoftAdapt variants.

    Attributes:
        epsilon: A float which is added to the denominator of a division for
          numerical stability.

    """

    def __init__(self):
        """Initializer of the base method."""
        self.epsilon = _EPSILON

    def _softmax(self,
                 input_tensor: torch.tensor,
                 beta: float = 1,
                 numerator_weights: torch.tensor = None,
                 shift_by_max_value: bool = True):
        """Implementation of SoftAdapt's modified softmax function.

        Args:
            input_tensor: A tensor of floats which will be used for computing
              the (modified) softmax function.
            beta: A float which is the scaling factor (as described in the
              manuscript).
            numerator_weights: A tensor of weights which are the actual value of
              of the loss components. This option is used for the
              "loss-weighted" variant of SoftAdapt.
            shift_by_max_value: A boolean indicating whether we want the values
              in the input tensor to be shifted by the maximum value.

        Returns:
            A tensor of floats that are the softmax results.

        Raises:
            None.

        """
        if shift_by_max_value:
            exp_of_input = torch.exp(beta * (input_tensor - input_tensor.max()))
        else:
            exp_of_input = torch.exp(beta * input_tensor)

        # This option will be used for the "loss-weighted" variant of SoftAdapt.
        if numerator_weights is not None:
            exp_of_input = torch.multiply(numerator_weights, exp_of_input)

        return exp_of_input / (torch.sum(exp_of_input) + self.epsilon)


    def _compute_rates_of_change(self,
                                 input_tensor:torch.tensor,
                                 order: int = 5,
                                 verbose: bool = True):
        """Base class method for computing loss functions rate of change.

        Args:
            input_tensor: A tensor of floats containing loss evaluations at the
              previous 'n' points (as many points as the order) of the finite
              difference method.
            order: An integer indicating the order of the finite difference
              method we want to use. The function will use the length of the
              'input_array' array if no values is provided.
            verbose: Whether we want the function to print out information about
              computations or not.

        Returns:
            The approximated derivative as a float value.

        Raises:
            None.

        """
        return _get_finite_difference(input_array = input_tensor.numpy(),
                                      order = order,
                                      verbose = verbose)
    
class LossWeightedSoftAdapt(SoftAdaptBase):
    """Class implementation of the loss-weighted SoftAdapt variant.

    The loss-weighted variant of SoftAdapt is described in section 3.1.1 of our
    manuscript (located at: https://arxiv.org/pdf/1912.12355.pdf).

    Attributes:
        beta: A float that is the 'beta' hyperparameter in our manuscript. If
          beta > 0, then softAdapt will pay more attention the worst performing
          loss component. If beta < 0, then SoftAdapt will assign higher weights
          to the better performing components. Beta==0 is the trivial case and
          all loss components will have coefficient 1.

        accuracy_order: An integer indicating the accuracy order of the finite
          volume approximation of each loss component's slope.
    """

    def __init__(self, beta: float = 0.1, accuracy_order: int = None):
        """SoftAdapt class initializer."""
        super().__init__()
        self.beta = beta
        # Passing "None" as the order of accuracy sets the highest possible
        # accuracy in the finite difference approximation.
        self.accuracy_order = accuracy_order

    def get_component_weights(self,
                               *loss_component_values: Tuple[torch.tensor],
                               verbose: bool = False):
        """Class method for SoftAdapt weights.

        Args:
            loss_component_values: A tuple consisting of the values of the each
              loss component that have been stored for the past 'n' iterations
              or epochs (as described in the manuscript).
            verbose: A boolean indicating user preference for whether internal
              functions should print out information and warning about
              computations.
        Returns:
            The computed weights for each loss components. For example, if there
            were 5 loss components, say (l_1, l_2, l_3, l_4, l_5), then the
            return tensor will be the weights (alpha_1, alpha_2, alpha_3,
            alpha_4, alpha_5) in the order of the loss components.

        Raises:
            None.

        """
        if len(loss_component_values) == 1:
            print("==> Warning: You have only passed on the values of one loss"
                  " component, which will result in trivial weighting.")

        rates_of_change = []
        average_loss_values = []

        for loss_points in loss_component_values:
            # Compute the rates of change for each one of the loss components.
            rates_of_change.append(
                self._compute_rates_of_change(loss_points,
                                              self.accuracy_order,
                                              verbose=verbose))
            average_loss_values.append(torch.mean(loss_points.float()))

        rates_of_change = torch.tensor(rates_of_change)
        average_loss_values = torch.tensor(average_loss_values)
        # Calculate the weight and return the values.
        return self._softmax(input_tensor=rates_of_change,
                             beta=self.beta,
                             numerator_weights = average_loss_values,
                             )
    
class NormalizedSoftAdapt(SoftAdaptBase):
    """The normalized-slopes variant class.

    The normalized variant of SoftAdapt is described in section 3.1.3 of our
    manuscript (located at: https://arxiv.org/pdf/1912.12355.pdf).

    Attributes:
        beta: A float that is the 'beta' hyperparameter in our manuscript. If
          beta > 0, then softAdapt will pay more attention the worst performing
          loss component. If beta < 0, then SoftAdapt will assign higher weights
          to the better performing components. Beta==0 is the trivial case and
          all loss components will have coefficient 1.

        accuracy_order: An integer indicating the accuracy order of the finite
          volume approximation of each loss component's slope.

    """

    def __init__(self, beta: float = 0.1, accuracy_order: int = None):
        """SoftAdapt class initializer."""
        super().__init__()
        self.beta = beta
        # Passing "None" as the order of accuracy sets the highest possible
        # accuracy in the finite difference approximation.
        self.accuracy_order = accuracy_order

    def get_component_weights(self,
                               *loss_component_values: Tuple[torch.tensor],
                               verbose: bool = False):
        """Class method for SoftAdapt weights.

        Args:
            loss_component_values: A tuple consisting of the values of the each
              loss component that have been stored for the past 'n' iterations
              or epochs (as described in the manuscript).
            verbose: A boolean indicating user preference for whether internal
              functions should print out information and warning about
              computations.
        Returns:
            The computed weights for each loss components. For example, if there
            were 5 loss components, say (l_1, l_2, l_3, l_4, l_5), then the
            return tensor will be the weights (alpha_1, alpha_2, alpha_3,
            alpha_4, alpha_5) in the order of the loss components.

        Raises:
            None.

        """
        if len(loss_component_values) == 1:
            print("==> Warning: You have only passed on the values of one loss"
                  " component, which will result in trivial weighting.")

        rates_of_change = []

        for loss_points in loss_component_values:
            # Compute the rates of change for each one of the loss components.
            rates_of_change.append(
                self._compute_rates_of_change(loss_points,
                                              self.accuracy_order,
                                              verbose=verbose))

        rates_of_change = torch.tensor(rates_of_change)/torch.sum(
                                                torch.tensor(rates_of_change))

        # Calculate the weight and return the values.
        return self._softmax(input_tensor=rates_of_change, beta=self.beta)
    
class SoftAdapt(SoftAdaptBase):
    """The original variant class.

    The original variant of SoftAdapt is described in section 3.1.1 of our
    manuscript (located at: https://arxiv.org/pdf/1912.12355.pdf).

    Attributes:
        beta: A float that is the 'beta' hyperparameter in our manuscript. If
          beta > 0, then softAdapt will pay more attention the worst performing
          loss component. If beta < 0, then SoftAdapt will assign higher weights
          to the better performing components. Beta==0 is the trivial case and
          all loss components will have coefficient 1.

        accuracy_order: An integer indicating the accuracy order of the finite
          volume approximation of each loss component's slope.

    """

    def __init__(self, beta: float = 0.1, accuracy_order: int = None):
        """SoftAdapt class initializer."""
        super().__init__()
        self.beta = beta
        # Passing "None" as the order of accuracy sets the highest possible
        # accuracy in the finite difference approximation.
        self.accuracy_order = accuracy_order

    def get_component_weights(self,
                               *loss_component_values: Tuple[torch.tensor],
                               verbose: bool = False):
        """Class method for SoftAdapt weights.

        Args:
            loss_component_values: A tuple consisting of the values of the each
              loss component that have been stored for the past 'n' iterations
              or epochs (as described in the manuscript).
            verbose: A boolean indicating user preference for whether internal
              functions should print out information and warning about
              computations.
        Returns:
            The computed weights for each loss components. For example, if there
            were 5 loss components, say (l_1, l_2, l_3, l_4, l_5), then the
            return tensor will be the weights (alpha_1, alpha_2, alpha_3,
            alpha_4, alpha_5) in the order of the loss components.

        Raises:
            None.

        """
        if len(loss_component_values) == 1:
            print("==> Warning: You have only passed on the values of one loss"
                  " component, which will result in trivial weighting.")

        rates_of_change = []

        for loss_points in loss_component_values:
            # Compute the rates of change for each one of the loss components.
            rates_of_change.append(
                self._compute_rates_of_change(loss_points,
                                              self.accuracy_order,
                                              verbose=verbose))

        rates_of_change = torch.tensor(rates_of_change)
        # Calculate the weight and return the values.
        return self._softmax(input_tensor=rates_of_change, beta=self.beta)