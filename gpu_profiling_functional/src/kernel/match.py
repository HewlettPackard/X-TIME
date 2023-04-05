#
#  Copyright (2023) Hewlett Packard Enterprise Development LP
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import numpy as np
from math import sqrt

import pycuda.autoinit
import pycuda.driver as drv

from pycuda.compiler import SourceModule

import os

CUDA_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "match.cu")
CUDA_kernel = "match"

with open(CUDA_file, "r") as file:
    CUDA_source = file.read()

kernel = SourceModule(CUDA_source)
cuda_match = kernel.get_function(CUDA_kernel)

def match(cam_input, cam_table, leaves):
    input_matrix = cam_input.flatten(order = "C").astype(np.float32)
    cam_table_matrix = cam_table.flatten(order = "C").astype(np.float32)

    output_vector = np.zeros(cam_input.shape[0]).astype(np.float32)
    leaves_vector = leaves.flatten(order = "C").astype(np.float32)

    features = cam_input.shape[1]

    max_block_size = 1024
    total_threads_needed = cam_input.shape[0] * cam_table.shape[0]

    optimal_thread_layout = (int(cam_input.shape[0]),
                             int(cam_table.shape[0]),
                             int(1))

    if total_threads_needed < max_block_size:
        blocks = int(1)
        block_size = optimal_thread_layout
        grid_size = (1, 1)
    else:
        blocks = int(np.ceil(((cam_input.shape[0] *
                               cam_table.shape[0]) +
                              max_block_size - 1) / max_block_size))

        block_size = (int(np.ceil(sqrt(max_block_size))),
                      int(np.ceil(sqrt(max_block_size))),
                      1)

        grid_size = (int(np.ceil(sqrt(blocks))),
                     int(np.ceil(sqrt(blocks))))

    cuda_match(drv.In(input_matrix),
               drv.In(cam_table_matrix),
               drv.InOut(output_vector),
               drv.In(leaves_vector),
               np.int64(features),
               np.int64(cam_input.shape[0]),
               np.int64(cam_table.shape[0]),
               block = block_size,
               grid = grid_size)

    return output_vector
