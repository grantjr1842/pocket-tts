//! Runtime CPU feature detection for SIMD operations
//!
//! This module provides runtime detection of CPU features to enable
//! optimal SIMD kernel selection and graceful fallback to scalar operations.

/// CPU feature flags for SIMD instruction sets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuFeatures {
    /// AVX2 (Advanced Vector Extensions 2) - 256-bit vectors
    pub has_avx2: bool,
    /// AVX-512 Foundation - 512-bit vectors
    pub has_avx512f: bool,
    /// AVX-512 Vector Length Extensions - variable vector lengths
    pub has_avx512vl: bool,
    /// Fused Multiply-Add operations
    pub has_fma: bool,
    /// SSE4.1 - Streaming SIMD Extensions 4.1
    pub has_sse41: bool,
    /// SSE4.2 - Streaming SIMD Extensions 4.2
    pub has_sse42: bool,
    /// ARM NEON support
    pub has_neon: bool,
    /// ARM Advanced SIMD (ASIMD)
    pub has_asimd: bool,
}

impl CpuFeatures {
    /// Detect available CPU features at runtime
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                has_avx2: is_x86_feature_detected!("avx2"),
                has_avx512f: is_x86_feature_detected!("avx512f"),
                has_avx512vl: is_x86_feature_detected!("avx512vl"),
                has_fma: is_x86_feature_detected!("fma"),
                has_sse41: is_x86_feature_detected!("sse4.1"),
                has_sse42: is_x86_feature_detected!("sse4.2"),
                has_neon: false,  // ARM only
                has_asimd: false, // ARM only
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            Self {
                has_avx2: false,     // x86 only
                has_avx512f: false,  // x86 only
                has_avx512vl: false, // x86 only
                has_fma: is_aarch64_feature_detected!("fma"),
                has_sse41: false, // x86 only
                has_sse42: false, // x86 only
                has_neon: is_aarch64_feature_detected!("neon"),
                has_asimd: is_aarch64_feature_detected!("asimd"),
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // Fallback for other architectures
            Self {
                has_avx2: false,
                has_avx512f: false,
                has_avx512vl: false,
                has_fma: false,
                has_sse41: false,
                has_sse42: false,
                has_neon: false,
                has_asimd: false,
            }
        }
    }

    /// Get the best vector width for f64 operations
    pub fn best_vector_width_f64(&self) -> usize {
        if self.has_avx512f {
            8 // AVX-512: 8 x f64 in 512-bit
        } else if self.has_avx2 {
            4 // AVX2: 4 x f64 in 256-bit
        } else if self.has_sse41 {
            2 // SSE: 2 x f64 in 128-bit
        } else {
            1 // Scalar
        }
    }

    /// Get the best vector width for f32 operations
    pub fn best_vector_width_f32(&self) -> usize {
        if self.has_avx512f {
            16 // AVX-512: 16 x f32 in 512-bit
        } else if self.has_avx2 {
            8 // AVX2: 8 x f32 in 256-bit
        } else if self.has_sse41 {
            4 // SSE: 4 x f32 in 128-bit
        } else {
            1 // Scalar
        }
    }

    /// Check if the CPU supports any SIMD instructions
    pub fn has_simd_support(&self) -> bool {
        self.has_avx2 || self.has_avx512f || self.has_sse41 || self.has_neon
    }

    /// Check if the CPU supports AVX2 or better
    pub fn has_avx2_or_better(&self) -> bool {
        self.has_avx2 || self.has_avx512f
    }

    /// Check if the CPU supports AVX-512
    pub fn has_avx512_support(&self) -> bool {
        self.has_avx512f
    }

    /// Get a human-readable description of CPU features
    pub fn description(&self) -> &'static str {
        if self.has_avx512f {
            "AVX-512"
        } else if self.has_avx2 {
            "AVX2"
        } else if self.has_sse41 {
            "SSE4.1"
        } else if self.has_neon {
            "NEON"
        } else {
            "Scalar"
        }
    }
}

impl Default for CpuFeatures {
    fn default() -> Self {
        Self::detect()
    }
}

/// Global CPU features cache
static mut CPU_FEATURES_CACHE: Option<CpuFeatures> = None;
static mut CPU_FEATURES_INIT: std::sync::Once = std::sync::Once::new();

/// Get cached CPU features (detects once and caches the result)
pub fn get_cpu_features() -> CpuFeatures {
    unsafe {
        CPU_FEATURES_INIT.call_once(|| {
            CPU_FEATURES_CACHE = Some(CpuFeatures::detect());
        });
        CPU_FEATURES_CACHE.unwrap_or_default()
    }
}

/// Reset the CPU features cache (useful for testing)
#[cfg(test)]
pub fn reset_cpu_features_cache() {
    unsafe {
        CPU_FEATURES_CACHE = None;
        CPU_FEATURES_INIT = std::sync::Once::new();
    }
}

/// CPU feature detection results for debugging
#[derive(Debug, Clone)]
pub struct CpuFeatureReport {
    pub features: CpuFeatures,
    pub architecture: String,
    pub vector_width_f64: usize,
    pub vector_width_f32: usize,
    pub has_simd_support: bool,
}

impl CpuFeatureReport {
    /// Generate a comprehensive CPU feature report
    pub fn generate() -> Self {
        let features = get_cpu_features();

        Self {
            architecture: std::env::consts::ARCH.to_string(),
            vector_width_f64: features.best_vector_width_f64(),
            vector_width_f32: features.best_vector_width_f32(),
            has_simd_support: features.has_simd_support(),
            features,
        }
    }

    /// Print the feature report
    pub fn print(&self) {
        println!("CPU Feature Report");
        println!("==================");
        println!("Architecture: {}", self.architecture);
        println!("SIMD Support: {}", self.has_simd_support);
        println!("Feature Level: {}", self.features.description());
        println!("Vector Width (f64): {}", self.vector_width_f64);
        println!("Vector Width (f32): {}", self.vector_width_f32);
        println!();
        println!("Detailed Features:");
        println!("  AVX2: {}", self.features.has_avx2);
        println!("  AVX-512F: {}", self.features.has_avx512f);
        println!("  AVX-512VL: {}", self.features.has_avx512vl);
        println!("  FMA: {}", self.features.has_fma);
        println!("  SSE4.1: {}", self.features.has_sse41);
        println!("  SSE4.2: {}", self.features.has_sse42);
        println!("  NEON: {}", self.features.has_neon);
        println!("  ASIMD: {}", self.features.has_asimd);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_features_detection() {
        reset_cpu_features_cache();
        let features = get_cpu_features();

        // Should not panic
        assert!(features.best_vector_width_f64() > 0);
        assert!(features.best_vector_width_f32() > 0);

        // Vector width for f32 should be >= f64
        assert!(features.best_vector_width_f32() >= features.best_vector_width_f64());
    }

    #[test]
    fn test_cpu_feature_report() {
        let report = CpuFeatureReport::generate();

        // Should not panic
        assert!(!report.architecture.is_empty());
        assert!(report.vector_width_f64 > 0);
        assert!(report.vector_width_f32 > 0);
    }

    #[test]
    fn test_cpu_features_cache() {
        reset_cpu_features_cache();

        let features1 = get_cpu_features();
        let features2 = get_cpu_features();

        // Should return the same cached result
        assert_eq!(features1.has_avx2, features2.has_avx2);
        assert_eq!(features1.has_avx512f, features2.has_avx512f);
    }

    #[test]
    fn test_feature_descriptions() {
        let features = get_cpu_features();

        let desc = features.description();
        assert!(!desc.is_empty());

        // Should be one of the expected descriptions
        let valid_descriptions = ["AVX-512", "AVX2", "SSE4.1", "NEON", "Scalar"];
        assert!(valid_descriptions.contains(&desc));
    }

    #[test]
    fn test_simd_support_detection() {
        let features = get_cpu_features();

        // SIMD support should be consistent with feature flags
        let has_simd = features.has_simd_support();
        assert_eq!(
            has_simd,
            features.has_avx2 || features.has_avx512f || features.has_sse41 || features.has_neon
        );
    }
}
