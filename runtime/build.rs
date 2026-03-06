fn main() {
    let os =
        std::env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");
    if os.as_str() != "linux" {
        panic!("Can only build on Linux")
    }
    let third_party = std::path::Path::new("third-party");
    if !third_party.exists() {
        panic!("Third-party deps missing! Run `just build-deps`.");
    }

    if let Some(libtorch_lib_path) = std::env::var_os("DEP_TCH_LIBTORCH_LIB") {
        println!(
            "cargo:rustc-link-arg=-Wl,-rpath={}",
            libtorch_lib_path.to_string_lossy()
        );
    }
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
    println!("cargo:rustc-link-arg=-ltorch");
}
