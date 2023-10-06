# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.ops._op_common_window import CommonWindow


class BlackmanWindow(CommonWindow):
    """
    Returns
    :math:`\\omega_n = 0.42 - 0.5 \\cos \\left( \\frac{2\\pi n}{N-1} \\right) +
    0.08 \\cos \\left( \\frac{4\\pi n}{N-1} \\right)`
    where *N* is the window length.
    See `blackman_window
    <https://pytorch.org/docs/stable/generated/torch.blackman_window.html>`_
    """

    def _run(self, size, output_datatype=None, periodic=None):  # type: ignore
        if periodic:
            window = np.blackman(size + 1)[:-1]
        else:
            window = np.blackman(size)

        return self._end(size, window, output_datatype)
