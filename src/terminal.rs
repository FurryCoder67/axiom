// ANSI color codes for terminal output
pub const CYAN: &str = "\x1b[36m";
pub const YELLOW: &str = "\x1b[33m";
pub const RED: &str = "\x1b[31m";
pub const GREEN: &str = "\x1b[32m";
pub const DIM: &str = "\x1b[2m";
pub const BOLD: &str = "\x1b[1m";
pub const MAGENTA: &str = "\x1b[35m";
pub const RESET: &str = "\x1b[0m";

pub fn cyan(msg: &str) -> String {
    format!("{}{}{}", CYAN, msg, RESET)
}

pub fn yellow(msg: &str) -> String {
    format!("{}{}{}", YELLOW, msg, RESET)
}

pub fn red(msg: &str) -> String {
    format!("{}{}{}", RED, msg, RESET)
}

pub fn green(msg: &str) -> String {
    format!("{}{}{}", GREEN, msg, RESET)
}

pub fn dim(msg: &str) -> String {
    format!("{}{}{}", DIM, msg, RESET)
}

pub fn bold(msg: &str) -> String {
    format!("{}{}{}", BOLD, msg, RESET)
}

/// Render a probability bar: [████░░░░] 0.72
pub fn score_bar(score: f64) -> String {
    let filled = (score * 8.0).round() as i32;
    let filled = filled.clamp(0, 8) as usize;
    let empty = 8 - filled;
    format!(
        "{}{}{}{} {:.2}",
        YELLOW,
        "█".repeat(filled),
        "░".repeat(empty),
        RESET,
        score
    )
}

/// Print the REPL prompt.
pub fn print_prompt() {
    print!("{}axiom ▶{} ", MAGENTA, RESET);
    use std::io::Write;
    let _ = std::io::stdout().flush();
}