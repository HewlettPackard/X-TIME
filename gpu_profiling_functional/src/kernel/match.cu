//
// Copyright (2023) Hewlett Packard Enterprise Development LP
//
// Licensed under the Apache License, Version 2.0 (the "License");
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

__global__ void match(float *input_matrix,
                      float *threshold_matrix,
                      float *output_vector,
                      float *leaves_vector,
                      int features,
                      long int input_lines,
                      long int threshold_lines) {
    float input;
    float min_threshold;
    float max_threshold;

    long int input_row;
    long int threshold_row;
    int match = 1;

    long int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    long int threadId = blockId * (blockDim.x * blockDim.y) +
        (threadIdx.y * blockDim.x) + threadIdx.x;

    input_row = threadId / threshold_lines;
    threshold_row = threadId % threshold_lines;

    if(threadId < input_lines * threshold_lines) {
        for(int i = 0; i < features; i++) {
            input = input_matrix[(input_row * features) + i];

            min_threshold = threshold_matrix[(threshold_row * (2 * features)) +
                                             (2 * i)];
            max_threshold = threshold_matrix[(threshold_row * (2 * features)) +
                                             ((2 * i) + 1)];

            if(not ((isnan(min_threshold) or input > min_threshold) and
                    (isnan(max_threshold) or input < max_threshold))) {
                match = 0;
                break;
            }
        }

        if(input_row < input_lines and threshold_row < threshold_lines and match == 1) {
            atomicAdd(&output_vector[input_row], leaves_vector[threshold_row]);
        }
    }
}
