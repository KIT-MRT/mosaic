from typing import cast

import numpy as np
import numpy.typing as npt


def rotation_matrix_2d(theta: float) -> npt.NDArray[np.float64]:
    c = cast(float, np.cos(theta))
    s = cast(float, np.sin(theta))
    return np.array([[c, -s], [s, c]])
