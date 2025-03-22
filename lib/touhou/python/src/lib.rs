use pyo3::prelude::*;
use pyo3::types::PyBytes;
use touhou_formats::th06::pbg3;
use std::fs::File;
use std::io::BufReader;

#[cfg(feature = "glide")]
mod glide;

#[pyclass]
struct PBG3 {
    inner: pbg3::PBG3<BufReader<File>>,
}

#[pymethods]
impl PBG3 {
    #[staticmethod]
    fn from_filename(filename: &str) -> PyResult<PBG3> {
        let inner = pbg3::from_path_buffered(filename)?;
        Ok(PBG3 {
            inner
        })
    }

    #[getter]
    fn file_list(&self) -> Vec<String> {
        self.inner.list_files().cloned().collect()
    }

    fn list_files(&self) -> Vec<String> {
        self.inner.list_files().cloned().collect()
    }

    fn get_file(&mut self, py: Python, name: &str) -> PyObject {
        let data = self.inner.get_file(name, true).unwrap();
        PyBytes::new(py, &data).into_py(py)
    }
}

#[pymodule]
fn libtouhou(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PBG3>()?;
    #[cfg(feature = "glide")]
    m.add_submodule(glide::module(py)?)?;
    Ok(())
}
