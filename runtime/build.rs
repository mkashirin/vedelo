use std::{env, path::PathBuf, process::Command};

fn main() {
    let os = env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");
    assert!(
        matches!(os.as_str(), "linux" | "macos"),
        "Only Linux and macOS are supported"
    );
    for name in [
        "LIBTORCH_LIB",
        "LIBTORCH",
        "LIBTORCH_USE_PYTORCH",
        "VIRTUAL_ENV",
        "PATH",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    // torch-sys is transitive: its DEP_TCH_* metadata does not reach us.
    // Match the pinned torch-sys interpreter choice (it does not read PYTHON).
    // Pinned torch-sys treats LIBTORCH_LIB as a root and appends /lib,
    // not as an actual split library directory. Avoid inconsistent linking.
    assert!(
        env::var_os("LIBTORCH_LIB").is_none(),
        "LIBTORCH_LIB split overrides are unsupported: unset it and use LIBTORCH=<root> (libraries in <root>/lib) or LIBTORCH_USE_PYTORCH=1"
    );
    let lib = if env::var_os("LIBTORCH_USE_PYTORCH").is_some() {
        let python = if env::var_os("VIRTUAL_ENV").is_some() {
            "python"
        } else {
            "python3"
        };
        let output = Command::new(python).args(["-c", "from torch.utils import cpp_extension; print(cpp_extension.library_paths()[-1])"])
            .output().expect("Unable to run torch-sys Python interpreter");
        assert!(
            output.status.success(),
            "PyTorch library discovery failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let path = String::from_utf8(output.stdout)
            .expect("PyTorch path is not UTF-8");
        assert!(!path.trim().is_empty(), "PyTorch returned no library path");
        PathBuf::from(path.trim())
    } else {
        PathBuf::from(env::var_os("LIBTORCH").expect("Set LIBTORCH to the libtorch root or LIBTORCH_USE_PYTORCH=1; implicit download/system discovery is not supported by this runtime" )).join("lib")
    };
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
    if os == "linux" {
        println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
    }
    println!("cargo:rustc-link-arg=-ltorch");
}
