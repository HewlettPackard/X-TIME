#![cfg(feature = "python")]

mod catboost;
mod treelite;

use std::{fmt, fs};

use numpy::{IntoPyArray, PyArray2};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

trait IntoPyResult<T, E> {
    fn into_py_result(self) -> PyResult<T>;
}

impl<T, E: fmt::Display> IntoPyResult<T, E> for Result<T, E> {
    fn into_py_result(self) -> PyResult<T> {
        self.map_err(|err| PyErr::new::<PyRuntimeError, _>(format!("{err}")))
    }
}

#[pymodule]
fn _xtimec(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn compile_treelite(py: Python<'_>, model_json: &str) -> PyResult<Py<PyArray2<f64>>> {
        let model = treelite::Model::from_json(model_json).into_py_result()?;
        let (compiled_model, shape) = treelite::compile_model(&model);
        Ok(compiled_model.into_pyarray(py).reshape(shape)?.to_owned())
    }

    #[pyfn(m)]
    fn compile_catboost(py: Python<'_>, path: &str) -> PyResult<Py<PyArray2<f64>>> {
        let json = fs::read_to_string(path).into_py_result()?;
        let model = catboost::Model::from_json(&json).into_py_result()?;
        let (compiled_model, shape) = catboost::compile_model(&model);
        Ok(compiled_model.into_pyarray(py).reshape(shape)?.to_owned())
    }

    m.add_function(wrap_pyfunction!(compile_treelite, m)?)?;
    m.add_function(wrap_pyfunction!(compile_catboost, m)?)?;

    Ok(())
}
