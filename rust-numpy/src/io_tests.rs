#[cfg(test)]
mod io_tests {
    use crate::io::{
        detect_file_format_from_filename, load, load_npz, save, savez, savez_compressed,
        savez_compressed_many, savez_many, FileFormat, MmapMode,
    };
    use crate::Array;
    use std::collections::HashMap;
    use std::fs;
    use std::path::Path;

    fn get_temp_path(name: &str) -> String {
        let temp_dir = std::env::temp_dir();
        temp_dir.join(name).to_string_lossy().to_string()
    }

    fn cleanup(path: &str) {
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_mmap_mode_parsing() {
        assert!("r".parse::<MmapMode>().unwrap() == MmapMode::Read);
        assert!("r+".parse::<MmapMode>().unwrap() == MmapMode::ReadWrite);
        assert!("c".parse::<MmapMode>().unwrap() == MmapMode::Read);
    }

    #[test]
    fn test_detect_file_format() {
        let result = detect_file_format_from_filename("test.npy");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), FileFormat::Npy);

        let result = detect_file_format_from_filename("test.npz");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), FileFormat::Npz);

        let result = detect_file_format_from_filename("test.txt");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), FileFormat::Text);
    }

    #[test]
    fn test_savez_and_load_npz() {
        let path = get_temp_path("test_savez.npz");
        cleanup(&path);

        // Create test arrays
        let a = Array::from_vec(vec![1i32, 2, 3, 4, 5]);
        let b = Array::from_vec(vec![10.0f64, 20.0, 30.0]);

        // Save as NPZ
        let result = savez(&path, vec![("array_a", &a), ("array_b", &b)]);
        assert!(result.is_ok(), "savez failed: {:?}", result.err());
        assert!(Path::new(&path).exists(), "NPZ file was not created");

        // Load back
        let loaded_arrays = load_npz::<i32>(&path).unwrap();
        assert!(loaded_arrays.contains_key("array_a"));
        assert!(!loaded_arrays.contains_key("array_b")); // Different type

        let loaded_a = loaded_arrays.get("array_a").unwrap();
        assert_eq!(loaded_a.to_vec(), vec![1, 2, 3, 4, 5]);

        cleanup(&path);
    }

    #[test]
    fn test_savez_compressed_and_load_npz() {
        let path = get_temp_path("test_savez_compressed.npz");
        cleanup(&path);

        // Create test arrays
        let a = Array::from_vec(vec![100i64, 200, 300]);
        let b = Array::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0]);

        // Save as compressed NPZ
        let result = savez_compressed(&path, vec![("matrix_a", &a), ("matrix_b", &b)]);
        assert!(result.is_ok(), "savez_compressed failed: {:?}", result.err());
        assert!(Path::new(&path).exists(), "Compressed NPZ file was not created");

        // Load back
        let loaded_arrays = load_npz::<i64>(&path).unwrap();
        assert!(loaded_arrays.contains_key("matrix_a"));

        let loaded_a = loaded_arrays.get("matrix_a").unwrap();
        assert_eq!(loaded_a.to_vec(), vec![100, 200, 300]);

        cleanup(&path);
    }

    #[test]
    fn test_savez_many_and_load_npz() {
        let path = get_temp_path("test_savez_many.npz");
        cleanup(&path);

        // Create test arrays
        let a = Array::from_vec(vec![1.0f64, 2.0, 3.0]);
        let b = Array::from_vec(vec![4.0f64, 5.0, 6.0]);
        let c = Array::from_vec(vec![7.0f64, 8.0, 9.0]);

        // Save multiple arrays with auto-generated names
        let result = savez_many(&path, &[&a, &b, &c]);
        assert!(result.is_ok(), "savez_many failed: {:?}", result.err());

        // Load back
        let loaded_arrays = load_npz::<f64>(&path).unwrap();
        assert!(loaded_arrays.contains_key("arr_0"));
        assert!(loaded_arrays.contains_key("arr_1"));
        assert!(loaded_arrays.contains_key("arr_2"));

        let loaded_a = loaded_arrays.get("arr_0").unwrap();
        let loaded_b = loaded_arrays.get("arr_1").unwrap();
        let loaded_c = loaded_arrays.get("arr_2").unwrap();

        assert_eq!(loaded_a.to_vec(), vec![1.0, 2.0, 3.0]);
        assert_eq!(loaded_b.to_vec(), vec![4.0, 5.0, 6.0]);
        assert_eq!(loaded_c.to_vec(), vec![7.0, 8.0, 9.0]);

        cleanup(&path);
    }

    #[test]
    fn test_savez_compressed_many() {
        let path = get_temp_path("test_savez_compressed_many.npz");
        cleanup(&path);

        // Create test arrays
        let a = Array::from_vec(vec![1i32, 2, 3]);
        let b = Array::from_vec(vec![4i32, 5, 6, 7]);

        // Save as compressed with auto-generated names
        let result = savez_compressed_many(&path, &[&a, &b]);
        assert!(
            result.is_ok(),
            "savez_compressed_many failed: {:?}",
            result.err()
        );

        // Load back
        let loaded_arrays = load_npz::<i32>(&path).unwrap();
        assert_eq!(loaded_arrays.len(), 2);
        assert!(loaded_arrays.contains_key("arr_0"));
        assert!(loaded_arrays.contains_key("arr_1"));

        cleanup(&path);
    }

    #[test]
    fn test_load_npz_empty_file() {
        let path = get_temp_path("test_empty.npz");
        cleanup(&path);

        // Create an empty valid ZIP file
        let file = fs::File::create(&path).unwrap();
        let mut zip = zip::ZipWriter::new(file);
        zip.finish().unwrap();

        // Try to load - should fail because no NPY files found
        let result = load_npz::<f64>(&path);
        assert!(result.is_err());

        cleanup(&path);
    }

    #[test]
    fn test_load_npz_nonexistent_file() {
        let path = get_temp_path("test_nonexistent.npz");
        cleanup(&path);

        let result = load_npz::<f64>(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_savez_with_2d_array() {
        let path = get_temp_path("test_2d.npz");
        cleanup(&path);

        // Create 2D array using from_shape_vec
        let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let arr = Array::from_shape_vec(vec![2, 3], data);

        // Save
        let result = savez(&path, vec![("matrix", &arr)]);
        assert!(result.is_ok(), "savez with 2D array failed: {:?}", result.err());

        // Load back
        let loaded_arrays = load_npz::<f64>(&path).unwrap();
        assert!(loaded_arrays.contains_key("matrix"));

        let loaded = loaded_arrays.get("matrix").unwrap();
        assert_eq!(loaded.shape(), vec![2, 3]);
        assert_eq!(loaded.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        cleanup(&path);
    }

    #[test]
    fn test_npz_roundtrip_multiple_types() {
        // Test with i8
        let path_i8 = get_temp_path("test_i8.npz");
        cleanup(&path_i8);
        let arr_i8 = Array::from_vec(vec![1i8, 2, 3]);
        savez(&path_i8, vec![("data", &arr_i8)]).unwrap();
        let loaded_i8 = load_npz::<i8>(&path_i8).unwrap();
        assert_eq!(loaded_i8.get("data").unwrap().to_vec(), vec![1i8, 2, 3]);
        cleanup(&path_i8);

        // Test with u32
        let path_u32 = get_temp_path("test_u32.npz");
        cleanup(&path_u32);
        let arr_u32 = Array::from_vec(vec![100u32, 200, 300]);
        savez(&path_u32, vec![("data", &arr_u32)]).unwrap();
        let loaded_u32 = load_npz::<u32>(&path_u32).unwrap();
        assert_eq!(loaded_u32.get("data").unwrap().to_vec(), vec![100u32, 200, 300]);
        cleanup(&path_u32);

        // Test with f32
        let path_f32 = get_temp_path("test_f32.npz");
        cleanup(&path_f32);
        let arr_f32 = Array::from_vec(vec![1.5f32, 2.5, 3.5]);
        savez(&path_f32, vec![("data", &arr_f32)]).unwrap();
        let loaded_f32 = load_npz::<f32>(&path_f32).unwrap();
        let loaded_data = loaded_f32.get("data").unwrap().to_vec();
        assert!((loaded_data[0] - 1.5).abs() < 0.001);
        assert!((loaded_data[1] - 2.5).abs() < 0.001);
        assert!((loaded_data[2] - 3.5).abs() < 0.001);
        cleanup(&path_f32);
    }
}
