fn main() {
    let os =
        std::env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");
    if os.as_str() != "linux" {
        // Do nothing for Windows and macOS.
    } else {
        if let Some(lib_path) = std::env::var_os("DEP_TCH_LIBTORCH_LIB") {
            println!(
                "cargo:rustc-link-arg=-Wl,-rpath={}",
                lib_path.to_string_lossy()
            );
        }
        println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
        println!("cargo:rustc-link-arg=-ltorch");
    }
}
