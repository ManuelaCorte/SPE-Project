from typing import Annotated, LiteralString

import numpy as np
import numpy.typing as npt

type DType = np.generic
type Float = np.float_
type Int = np.int_

type Matrix[S: LiteralString, DT: DType] = Annotated[npt.NDArray[DT], S]
