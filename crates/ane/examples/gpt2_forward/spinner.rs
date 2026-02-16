use std::io::{self, Write};
use std::time::Instant;

const FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

pub struct Spinner {
    message: String,
    start: Instant,
    frame: usize,
}

impl Spinner {
    pub fn new(message: &str) -> Self {
        let spinner = Self {
            message: message.to_string(),
            start: Instant::now(),
            frame: 0,
        };
        spinner.draw();
        spinner
    }

    pub fn update(&mut self, message: &str) {
        self.message = message.to_string();
        self.frame = (self.frame + 1) % FRAMES.len();
        self.draw();
    }

    fn draw(&self) {
        let frame = FRAMES[self.frame % FRAMES.len()];
        eprint!("\r\x1b[2K\x1b[1;36m{frame}\x1b[0m {}", self.message);
        let _ = io::stderr().flush();
    }

    pub fn finish(self, message: &str) {
        let elapsed = self.start.elapsed().as_secs_f64();
        eprint!("\r\x1b[2K\x1b[1;32m✓\x1b[0m {message} ({elapsed:.1}s)\n");
        let _ = io::stderr().flush();
    }
}
