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
import typing as t
from unittest import TestCase

import ray.tune.search.sample as sample
import yaml

from xtime.contrib.tune_ext import (
    Analysis,
    RandomVarDomain,
    RayTuneDriverToMLflowLoggerCallback,
    add_representers,
    gpu_available,
)


class TestYamlRepresenters(TestCase):
    """
    Here, `rv` stands for random variable.
    """

    def setUp(self) -> None:
        add_representers()

    def test_integer_domain(self) -> None:
        """Test serialization/deserialization for integer domain (`Integer`)."""
        self._test_uniform_samplers(sample.Integer)

    def test_float_domain(self) -> None:
        """Test serialization/deserialization for floating point domain (`Float`)."""
        self._test_uniform_samplers(sample.Float)
        self._test_normal_samplers()

    def test_categorical_domain(self) -> None:
        """Test serialization/deserialization for categorical domain (`Categorical`)."""
        cats = ["Garfield", "Tom Cat", "Puss in Boots"]

        def _check(_rv: sample.Categorical, _sampler_t: t.Optional[t.Type]) -> None:
            self.assertIsInstance(_rv, sample.Categorical)
            self.assertListEqual(cats, _rv.categories)
            self._check_sampler(type(_rv), _rv.sampler, sampler_type=_sampler_t)

        _check(_rv=self._dump_load(sample.choice(cats)), _sampler_t=sample.Categorical._Uniform)
        _check(_rv=self._dump_load(sample.Categorical(cats)), _sampler_t=None)
        _check(_rv=self._dump_load(sample.Categorical(cats).uniform()), _sampler_t=sample.Categorical._Uniform)
        _check(_rv=self._dump_load(sample.Categorical(cats).grid()), _sampler_t=sample.Grid)

    def _dump_load(self, rv: RandomVarDomain) -> RandomVarDomain:
        """Serialize and then deserialize random variable specification."""
        serialized_rv: str = yaml.dump(rv, Dumper=yaml.SafeDumper)
        self.assertIsInstance(serialized_rv, str)

        deserialized_rv = yaml.load(serialized_rv, Loader=yaml.SafeLoader)
        self.assertIsInstance(deserialized_rv, type(rv))

        return deserialized_rv

    def _check_object_type(self, obj: t.Any, expected_type: t.Optional[t.Type]) -> None:
        if expected_type is None:
            self.assertIsNone(obj)
        else:
            self.assertIsInstance(obj, expected_type)

    def _test_uniform_samplers(self, domain_t: t.Union[t.Type[sample.Integer], t.Type[sample.Float]]) -> None:
        lower, upper, base, q = (10, 100, 6, 2)

        self.assertIn(domain_t, {sample.Integer, sample.Float}, "Invalid domain type.")

        def _check(
            _rv: sample.Domain, _sampler_t: t.Optional[t.Type], _q: t.Optional = None, _base: t.Optional = None
        ) -> None:
            self.assertIsInstance(_rv, domain_t)
            self.assertEqual(lower, _rv.lower)
            self.assertEqual(upper, _rv.upper)
            self._check_sampler(type(_rv), _rv.sampler, sampler_type=_sampler_t, q=_q, base=_base)

        _check(self._dump_load(domain_t(lower, upper)), None, None, None)
        _check(self._dump_load(domain_t(lower, upper).uniform()), domain_t._Uniform, None, None)
        _check(self._dump_load(domain_t(lower, upper).loguniform(base)), domain_t._LogUniform, None, base)
        _check(self._dump_load(domain_t(lower, upper).uniform().quantized(q)), domain_t._Uniform, q, None)
        _check(self._dump_load(domain_t(lower, upper).loguniform(base).quantized(q)), domain_t._LogUniform, q, base)

        if domain_t is sample.Integer:
            random, log_random, q_random, q_log_random = (
                sample.randint,
                sample.lograndint,
                sample.qrandint,
                sample.qlograndint,
            )
        else:
            random, log_random, q_random, q_log_random = (
                sample.uniform,
                sample.loguniform,
                sample.quniform,
                sample.qloguniform,
            )

        _check(self._dump_load(random(lower, upper)), domain_t._Uniform, None, None)
        _check(self._dump_load(log_random(lower, upper, base)), domain_t._LogUniform, None, base)
        _check(self._dump_load(q_random(lower, upper, q)), domain_t._Uniform, q, None)
        _check(self._dump_load(q_log_random(lower, upper, q, base)), domain_t._LogUniform, q, base)

    def _test_normal_samplers(self) -> None:
        domain_t = sample.Float
        mean, sd = 1.1, 0.45
        q = 0.1

        def _check(_rv: sample.Float, _sampler_t: t.Optional[t.Type], _q: t.Optional = None) -> None:
            self.assertEqual(float("-inf"), _rv.lower)
            self.assertEqual(float("+inf"), _rv.upper)
            self._check_sampler(type(_rv), _rv.sampler, sampler_type=_sampler_t, q=_q, mean=mean, sd=sd)

        _check(self._dump_load(domain_t(None, None).normal(mean, sd)), domain_t._Normal, None)
        _check(self._dump_load(domain_t(None, None).normal(mean, sd).quantized(q)), domain_t._Normal, q)

        _check(self._dump_load(sample.randn(mean, sd)), domain_t._Normal, None)
        _check(self._dump_load(sample.qrandn(mean, sd, q)), domain_t._Normal, q)

    def _check_sampler(
        self,
        rv_type: t.Type,
        sampler: t.Optional[sample.Sampler],
        sampler_type: t.Optional[t.Type] = None,
        base: t.Optional[int] = None,
        q: t.Optional[int] = None,
        mean: t.Optional[float] = None,
        sd: t.Optional[float] = None,
    ) -> None:
        if q is not None:
            #
            self.assertIsNotNone(sampler_type, "Quantized sampler requires another sampler.")
            #
            self.assertIsInstance(sampler, sample.Quantized)
            self.assertEqual(q, sampler.q)
            sampler = sampler.sampler

        self._check_object_type(sampler, sampler_type)

        if rv_type is sample.Categorical or sampler is None or isinstance(sampler, sample.Uniform):
            return

        if isinstance(sampler, sample.LogUniform):
            self.assertEqual(base, sampler.base)
            return

        if isinstance(sampler, sample.Normal):
            self.assertEqual(mean, sampler.mean)
            self.assertEqual(sd, sampler.sd)
            return

        self.assertTrue(False, f"Unexpected sampler: {sampler}.")
