//! Test utilities for validating Rust NumPy implementation
//!
//! This module provides utilities to load and validate test cases
//! against the comprehensive NumPy test examples.

use crate::Array;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Deserialize, Serialize)]
pub struct TestCase {
    pub function: String,
    pub input: serde_json::Value,
    pub output: TestOutput,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TestOutput {
    pub data: Option<Vec<serde_json::Value>>,
    pub shape: Option<Vec<usize>>,
    pub dtype: Option<String>,
    pub value: Option<serde_json::Value>,
    pub description: Option<String>,
    pub result: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Clone)]
pub struct TestArray {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
    pub dtype: String,
}

impl TestArray {
    pub fn from_json(input: &serde_json::Value) -> Result<Self, Box<dyn std::error::Error>> {
        match input {
            serde_json::Value::Array(arr) => {
                let data: Vec<f64> = arr.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect();
                let len = data.len();
                Ok(TestArray {
                    data,
                    shape: vec![len],
                    dtype: "float64".to_string(),
                })
            }
            serde_json::Value::Object(obj) => {
                if let Some(object) = obj.get("object") {
                    Self::from_json(object)
                } else if let Some(x) = obj.get("x") {
                    Self::from_json(x)
                } else if let Some(y) = obj.get("y") {
                    Self::from_json(y)
                } else {
                    Err("Invalid input format".into())
                }
            }
            _ => Err("Unsupported input format".into()),
        }
    }

    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self, String> {
        assert_eq!(
            self.data.len(),
            new_shape.iter().product::<usize>(),
            "Shape mismatch"
        );
        Ok(TestArray {
            data: self.data.clone(),
            shape: new_shape.to_vec(),
            dtype: self.dtype.clone(),
        })
    }
}

pub struct NumPyTestValidator {
    test_data_dir: PathBuf,
}

impl NumPyTestValidator {
    pub fn new() -> Self {
        Self {
            test_data_dir: PathBuf::from("../test_data"), // Adjust path as needed
        }
    }

    pub fn with_path(path: &str) -> Self {
        Self {
            test_data_dir: PathBuf::from(path),
        }
    }

    pub fn load_test_cases(
        &self,
        category: &str,
    ) -> Result<Vec<TestCase>, Box<dyn std::error::Error>> {
        let file_path = self
            .test_data_dir
            .join(format!("test_cases_{}.json", category));
        let file = std::fs::File::open(file_path)?;
        let test_cases: Vec<TestCase> = serde_json::from_reader(file)?;
        Ok(test_cases)
    }

    pub fn validate_array_creation(&self) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let test_cases = self.load_test_cases("array_creation")?;
        let mut results = Vec::new();

        for test_case in test_cases {
            let result = match test_case.function.as_str() {
                "array" => self.test_array_creation(&test_case),
                "zeros" => self.test_zeros(&test_case),
                "ones" => self.test_ones(&test_case),
                "arange" => self.test_arange(&test_case),
                "linspace" => self.test_linspace(&test_case),
                "eye" => self.test_eye(&test_case),
                _ => Ok(format!("Skipped unknown function: {}", test_case.function)),
            };

            match result {
                Ok(msg) => results.push(msg),
                Err(e) => results.push(format!("FAILED: {}", e)),
            }
        }

        Ok(results)
    }

    pub fn validate_arithmetic(&self) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let test_cases = self.load_test_cases("arithmetic")?;
        let mut results = Vec::new();

        for test_case in test_cases {
            let result = match test_case.function.as_str() {
                "add" => self.test_add(&test_case),
                "subtract" => self.test_subtract(&test_case),
                "multiply" => self.test_multiply(&test_case),
                "divide" => self.test_divide(&test_case),
                "power" => self.test_power(&test_case),
                _ => Ok(format!("Skipped unknown function: {}", test_case.function)),
            };

            match result {
                Ok(msg) => results.push(msg),
                Err(e) => results.push(format!("FAILED: {}", e)),
            }
        }

        Ok(results)
    }

    pub fn validate_mathematical(&self) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let test_cases = self.load_test_cases("mathematical")?;
        let mut results = Vec::new();

        for test_case in test_cases {
            let result = match test_case.function.as_str() {
                "sqrt" => self.test_sqrt(&test_case),
                "abs" => self.test_abs(&test_case),
                "sin" => self.test_sin(&test_case),
                "cos" => self.test_cos(&test_case),
                "exp" => self.test_exp(&test_case),
                "log" => self.test_log(&test_case),
                _ => Ok(format!("Skipped unknown function: {}", test_case.function)),
            };

            match result {
                Ok(msg) => results.push(msg),
                Err(e) => results.push(format!("FAILED: {}", e)),
            }
        }

        Ok(results)
    }

    fn test_array_creation(
        &self,
        test_case: &TestCase,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let input_array = TestArray::from_json(&test_case.input)?;
        let expected = &test_case.output;

        // Call your Rust implementation
        let result = crate::array::array(input_array.data)?;

        // Validate shape
        if let Some(expected_shape) = &expected.shape {
            if result.shape() != expected_shape.as_slice() {
                return Err(format!(
                    "Shape mismatch: expected {:?}, got {:?}",
                    expected_shape,
                    result.shape()
                )
                .into());
            }
        }

        // Validate data
        if let Some(expected_data) = &expected.data {
            let result_data: Vec<f64> = result.iter().map(|&x| x).collect();
            let expected_vec: Vec<f64> = expected_data
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0))
                .collect();

            if !self.arrays_close(&result_data, &expected_vec) {
                return Err("Data mismatch".into());
            }
        }

        Ok(format!("✅ {}: PASSED", test_case.function))
    }

    fn test_zeros(&self, test_case: &TestCase) -> Result<String, Box<dyn std::error::Error>> {
        let input = &test_case.input;
        let shape: Vec<usize> = input["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();

        let result = Array::zeros(&shape);
        let expected = &test_case.output;

        // Validate shape
        if let Some(expected_shape) = &expected.shape {
            if result.shape() != expected_shape.as_slice() {
                return Err(format!(
                    "Shape mismatch: expected {:?}, got {:?}",
                    expected_shape,
                    result.shape()
                )
                .into());
            }
        }

        // Validate all zeros
        if !result.iter().all(|&x| x == 0.0) {
            return Err("Array contains non-zero values".into());
        }

        Ok(format!("✅ {}: PASSED", test_case.function))
    }

    fn test_ones(&self, test_case: &TestCase) -> Result<String, Box<dyn std::error::Error>> {
        let input = &test_case.input;
        let shape: Vec<usize> = input["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();

        let result = Array::ones(&shape);
        let expected = &test_case.output;

        // Validate shape
        if let Some(expected_shape) = &expected.shape {
            if result.shape() != expected_shape.as_slice() {
                return Err(format!(
                    "Shape mismatch: expected {:?}, got {:?}",
                    expected_shape,
                    result.shape()
                )
                .into());
            }
        }

        // Validate all ones
        if !result.iter().all(|&x| x == 1.0) {
            return Err("Array contains non-one values".into());
        }

        Ok(format!("✅ {}: PASSED", test_case.function))
    }

    fn test_arange(&self, test_case: &TestCase) -> Result<String, Box<dyn std::error::Error>> {
        let input = &test_case.input;
        let start = input["start"].as_i64().unwrap() as isize;
        let stop = input["stop"].as_i64().unwrap() as isize;
        let step = input
            .get("step")
            .map(|v| v.as_i64().unwrap() as isize)
            .unwrap_or(1);

        let result = Array::arange(start, stop, step);
        let expected = &test_case.output;

        // Validate shape
        if let Some(expected_shape) = &expected.shape {
            if result.shape() != expected_shape.as_slice() {
                return Err(format!(
                    "Shape mismatch: expected {:?}, got {:?}",
                    expected_shape,
                    result.shape()
                )
                .into());
            }
        }

        // Validate data
        if let Some(expected_data) = &expected.data {
            let result_data: Vec<f64> = result.iter().map(|&x| x).collect();
            let expected_vec: Vec<f64> = expected_data
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0))
                .collect();

            if !self.arrays_close(&result_data, &expected_vec) {
                return Err("Data mismatch".into());
            }
        }

        Ok(format!("✅ {}: PASSED", test_case.function))
    }

    fn test_linspace(&self, test_case: &TestCase) -> Result<String, Box<dyn std::error::Error>> {
        let input = &test_case.input;
        let start = input["start"].as_f64().unwrap();
        let stop = input["stop"].as_f64().unwrap();
        let num = input["num"].as_u64().unwrap() as usize;

        let result = Array::linspace(start, stop, num);
        let expected = &test_case.output;

        // Validate shape
        if let Some(expected_shape) = &expected.shape {
            if result.shape() != expected_shape.as_slice() {
                return Err(format!(
                    "Shape mismatch: expected {:?}, got {:?}",
                    expected_shape,
                    result.shape()
                )
                .into());
            }
        }

        // Validate data
        if let Some(expected_data) = &expected.data {
            let result_data: Vec<f64> = result.iter().map(|&x| x).collect();
            let expected_vec: Vec<f64> = expected_data
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0))
                .collect();

            if !self.arrays_close(&result_data, &expected_vec) {
                return Err("Data mismatch".into());
            }
        }

        Ok(format!("✅ {}: PASSED", test_case.function))
    }

    fn test_eye(&self, test_case: &TestCase) -> Result<String, Box<dyn std::error::Error>> {
        let input = &test_case.input;
        let n = input["N"].as_u64().unwrap() as usize;

        let result = Array::eye(n);
        let expected = &test_case.output;

        // Validate shape
        if let Some(expected_shape) = &expected.shape {
            if result.shape() != expected_shape.as_slice() {
                return Err(format!(
                    "Shape mismatch: expected {:?}, got {:?}",
                    expected_shape,
                    result.shape()
                )
                .into());
            }
        }

        // Validate data
        if let Some(expected_data) = &expected.data {
            let result_data: Vec<f64> = result.iter().map(|&x| x).collect();
            let expected_vec: Vec<f64> = expected_data
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0))
                .collect();

            if !self.arrays_close(&result_data, &expected_vec) {
                return Err("Data mismatch".into());
            }
        }

        Ok(format!("✅ {}: PASSED", test_case.function))
    }

    fn test_add(&self, test_case: &TestCase) -> Result<String, Box<dyn std::error::Error>> {
        let x_array = TestArray::from_json(&test_case.input["x"])?;
        let y_array = TestArray::from_json(&test_case.input["y"])?;

        // Call your implementation
        let result = crate::functions::add(&x_array.data, &y_array.data)?;

        // Validate
        let expected_data: Vec<f64> = test_case
            .output
            .data
            .as_ref()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0))
            .collect();

        if result.len() != expected_data.len() {
            return Err("Length mismatch".into());
        }

        for (actual, expected) in result.iter().zip(expected_data.iter()) {
            if (actual - expected).abs() > 1e-10 {
                return Err(format!("Value mismatch: {} != {}", actual, expected).into());
            }
        }

        Ok("✅ add: PASSED".to_string())
    }

    fn test_subtract(&self, test_case: &TestCase) -> Result<String, Box<dyn std::error::Error>> {
        let x_array = TestArray::from_json(&test_case.input["x"])?;
        let y_array = TestArray::from_json(&test_case.input["y"])?;

        let result = crate::functions::subtract(&x_array.data, &y_array.data)?;
        let expected_data: Vec<f64> = test_case
            .output
            .data
            .as_ref()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0))
            .collect();

        if !self.arrays_close(&result, &expected_data) {
            return Err("Data mismatch".into());
        }

        Ok("✅ subtract: PASSED".to_string())
    }

    fn test_multiply(&self, test_case: &TestCase) -> Result<String, Box<dyn std::error::Error>> {
        let x_array = TestArray::from_json(&test_case.input["x"])?;
        let y_array = TestArray::from_json(&test_case.input["y"])?;

        let result = crate::functions::multiply(&x_array.data, &y_array.data)?;
        let expected_data: Vec<f64> = test_case
            .output
            .data
            .as_ref()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0))
            .collect();

        if !self.arrays_close(&result, &expected_data) {
            return Err("Data mismatch".into());
        }

        Ok("✅ multiply: PASSED".to_string())
    }

    fn test_divide(&self, test_case: &TestCase) -> Result<String, Box<dyn std::error::Error>> {
        let x_array = TestArray::from_json(&test_case.input["x"])?;
        let y_array = TestArray::from_json(&test_case.input["y"])?;

        let result = crate::functions::divide(&x_array.data, &y_array.data)?;
        let expected_data: Vec<f64> = test_case
            .output
            .data
            .as_ref()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0))
            .collect();

        if !self.arrays_close(&result, &expected_data) {
            return Err("Data mismatch".into());
        }

        Ok("✅ divide: PASSED".to_string())
    }

    fn test_power(&self, test_case: &TestCase) -> Result<String, Box<dyn std::error::Error>> {
        let x_array = TestArray::from_json(&test_case.input["x"])?;
        let power = test_case.input["y"].as_i64().unwrap() as f64;

        let result = crate::functions::power(&x_array.data, power)?;
        let expected_data: Vec<f64> = test_case
            .output
            .data
            .as_ref()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0))
            .collect();

        if !self.arrays_close(&result, &expected_data) {
            return Err("Data mismatch".into());
        }

        Ok("✅ power: PASSED".to_string())
    }

    fn test_sqrt(&self, test_case: &TestCase) -> Result<String, Box<dyn std::error::Error>> {
        let x_array = TestArray::from_json(&test_case.input["x"])?;

        let result = crate::functions::sqrt(&x_array.data)?;
        let expected_data: Vec<f64> = test_case
            .output
            .data
            .as_ref()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0))
            .collect();

        if !self.arrays_close(&result, &expected_data) {
            return Err("Data mismatch".into());
        }

        Ok("✅ sqrt: PASSED".to_string())
    }

    fn test_abs(&self, test_case: &TestCase) -> Result<String, Box<dyn std::error::Error>> {
        let x_array = TestArray::from_json(&test_case.input["x"])?;

        let result = crate::functions::abs(&x_array.data)?;
        let expected_data: Vec<f64> = test_case
            .output
            .data
            .as_ref()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0))
            .collect();

        if !self.arrays_close(&result, &expected_data) {
            return Err("Data mismatch".into());
        }

        Ok("✅ abs: PASSED".to_string())
    }

    fn test_sin(&self, test_case: &TestCase) -> Result<String, Box<dyn std::error::Error>> {
        let x_array = TestArray::from_json(&test_case.input["x"])?;

        let result = crate::functions::sin(&x_array.data)?;
        let expected_data: Vec<f64> = test_case
            .output
            .data
            .as_ref()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0))
            .collect();

        if !self.arrays_close(&result, &expected_data) {
            return Err("Data mismatch".into());
        }

        Ok("✅ sin: PASSED".to_string())
    }

    fn test_cos(&self, test_case: &TestCase) -> Result<String, Box<dyn std::error::Error>> {
        let x_array = TestArray::from_json(&test_case.input["x"])?;

        let result = crate::functions::cos(&x_array.data)?;
        let expected_data: Vec<f64> = test_case
            .output
            .data
            .as_ref()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0))
            .collect();

        if !self.arrays_close(&result, &expected_data) {
            return Err("Data mismatch".into());
        }

        Ok("✅ cos: PASSED".to_string())
    }

    fn test_exp(&self, test_case: &TestCase) -> Result<String, Box<dyn std::error::Error>> {
        let x_array = TestArray::from_json(&test_case.input["x"])?;

        let result = crate::functions::exp(&x_array.data)?;
        let expected_data: Vec<f64> = test_case
            .output
            .data
            .as_ref()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0))
            .collect();

        if !self.arrays_close(&result, &expected_data) {
            return Err("Data mismatch".into());
        }

        Ok("✅ exp: PASSED".to_string())
    }

    fn test_log(&self, test_case: &TestCase) -> Result<String, Box<dyn std::error::Error>> {
        let x_array = TestArray::from_json(&test_case.input["x"])?;

        let result = crate::functions::log(&x_array.data)?;
        let expected_data: Vec<f64> = test_case
            .output
            .data
            .as_ref()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0))
            .collect();

        if !self.arrays_close(&result, &expected_data) {
            return Err("Data mismatch".into());
        }

        Ok("✅ log: PASSED".to_string())
    }

    fn arrays_close(&self, a: &[f64], b: &[f64]) -> bool {
        if a.len() != b.len() {
            return false;
        }

        const TOLERANCE: f64 = 1e-10;
        a.iter()
            .zip(b.iter())
            .all(|(&x, &y)| (x - y).abs() < TOLERANCE)
    }
}
