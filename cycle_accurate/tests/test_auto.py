###
# Copyright (2023) Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###

import sys
import itertools
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "tests")))
from utils import *

arg = readAutoArg()

parameters = readJson(arg['hyperparam'])

parsedParameters = parseParameter(parameters)

listConfiguration = list(itertools.product(*parsedParameters.values()))
numTotalConfig = len(listConfiguration)
for i, listConfig in enumerate(listConfiguration):
    print("\nRunning #{:d} configuration".format(i+1))
    updateConfig(arg['config'], parsedParameters, listConfig)
    runSST(arg)
    print("(#{:d} done) {:.1f}%".format(i+1, (i+1)/numTotalConfig*100))

calculatePower(arg)