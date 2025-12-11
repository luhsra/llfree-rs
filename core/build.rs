#![allow(unused_imports)]
use std::env;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread::available_parallelism;

use build_rs::*;

fn main() {
    #[allow(unused_variables)]
    let cores = available_parallelism().unwrap_or(NonZeroUsize::MIN).get();

    #[cfg(feature = "llc")]
    if input::cargo_feature("llc") {
        let root = input::cargo_manifest_dir();
        let llc_dir = env::var("LLC_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| root.join("../llc"));
        let Ok(llc_dir) = llc_dir.canonicalize() else {
            panic!("Path not found: {llc_dir:?}");
        };
        let outdir = input::out_dir();

        let bindings = bindgen::Builder::default()
            .header(llc_dir.join("include/llfree.h").to_string_lossy())
            .clang_arg(format!("-I{}", llc_dir.join("include").display()))
            .clang_arg(format!("-I{}", llc_dir.join("std").display()))
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .allowlist_item("(llf|LLF).*")
            .generate_cstr(true)
            .fit_macro_constants(true)
            .generate()
            .expect("Unable to generate bindings");

        bindings
            .write_to_file(outdir.join("bindings.rs"))
            .expect("Unable to write bindings");

        // Build the C project
        let is_debug = input::profile() == "debug";
        let output = Command::new("make")
            .arg(format!("DEBUG={}", is_debug as usize))
            .arg(format!("BUILDDIR={}", outdir.display()))
            .arg(format!("-j{cores}"))
            .arg("-C")
            .arg(&llc_dir)
            .output()
            .expect("Build llc library");

        if !output.status.success() {
            panic!(
                "Build llc library: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        // Link the C project static library
        output::rustc_link_search_kind("native", outdir);
        output::rustc_link_lib("llc");

        // Re-run the build script if any C source files change
        output::rerun_if_changed(llc_dir.join("Makefile"));
        output::rerun_if_changed(llc_dir.join("src"));
        output::rerun_if_changed(llc_dir.join("include"));
        output::rerun_if_changed(llc_dir.join("std"));
    }
}
