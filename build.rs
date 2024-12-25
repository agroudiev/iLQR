use std::process::Command;
pub fn main() {
    let python_inline_script =
        "from distutils import sysconfig;print(sysconfig.get_config_var('LIBDIR'))";
    let output = Command::new("python")
        .arg("-c")
        .arg(python_inline_script)
        .output()
        .expect("Failed to execute command");
    if !output.status.success() {
        panic!("Python command failed");
    }
    let python_lib_path = String::from_utf8_lossy(&output.stdout);
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", python_lib_path);
}
