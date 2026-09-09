fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), cerebro_tidex::BrainError> {
    cerebro_tidex::universal_shadow_runner::run_universal_shadow_runner()
}
