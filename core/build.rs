use std::{env, process::Command};

fn main() {
    if env::var("CARGO_FEATURE_LLC").is_ok() {
        let root = env::var("CARGO_MANIFEST_DIR").unwrap();
        let c_project_dir = format!("{root}/../llc");
        let outdir = env::var("OUT_DIR").unwrap();
        let is_debug = env::var("PROFILE").unwrap() == "debug";

        // Build the C project
        let output = Command::new("make")
            .arg(format!("DEBUG={}", is_debug as usize))
            .arg(format!("BUILDDIR={outdir}"))
            .arg("-C")
            .arg(&c_project_dir)
            .arg(format!("{outdir}/libllc.a"))
            .output()
            .expect("Failed to build C project using Makefile");

        if !output.status.success() {
            panic!(
                "Failed to build C project: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        // Link the C project static library
        println!("cargo:rustc-link-search=native={outdir}");

        // Re-run the build script if any C source files change
        println!("cargo:rerun-if-changed={c_project_dir}/Makefile");
        println!("cargo:rerun-if-changed={c_project_dir}/src");
        println!("cargo:rerun-if-changed={c_project_dir}/include");
        println!("cargo:rerun-if-changed={c_project_dir}/std");
    }
}
