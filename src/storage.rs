use crate::neural_net::NeuralNet;
use crate::types::TaskRecord;
use crate::Config;
use std::path::PathBuf;

/// Expand ~ to $HOME and resolve the data directory path.
pub fn data_dir_path(config: &Config) -> PathBuf {
    let dir = &config.agent.data_dir;
    let expanded = if dir.starts_with("~/") {
        if let Ok(home) = std::env::var("HOME") {
            format!("{}/{}", home, &dir[2..])
        } else {
            dir.clone()
        }
    } else {
        dir.clone()
    };
    PathBuf::from(expanded)
}

/// Ensure the data directory exists, return its path.
pub fn ensure_data_dir(config: &Config) -> PathBuf {
    let dir = data_dir_path(config);
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// Save neural net weights to disk as JSON.
pub fn save_weights(net: &NeuralNet, config: &Config) {
    let dir = ensure_data_dir(config);
    let path = dir.join("weights.json");
    if let Ok(json) = serde_json::to_string_pretty(net) {
        let _ = std::fs::write(&path, json);
    }
}

/// Load neural net weights from disk. Returns None if not found or invalid.
pub fn load_weights(config: &Config) -> Option<NeuralNet> {
    let dir = data_dir_path(config);
    let path = dir.join("weights.json");
    let data = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&data).ok()
}

/// Save task history as JSONL (one record per line).
pub fn save_history(history: &[TaskRecord], config: &Config) {
    let dir = ensure_data_dir(config);
    let path = dir.join("history.jsonl");
    let mut content = String::new();
    for record in history {
        if let Ok(json) = serde_json::to_string(record) {
            content.push_str(&json);
            content.push('\n');
        }
    }
    let _ = std::fs::write(&path, content);
}

/// Load task history from JSONL.
pub fn load_history(config: &Config) -> Vec<TaskRecord> {
    let dir = data_dir_path(config);
    let path = dir.join("history.jsonl");
    let mut history = Vec::new();
    if let Ok(content) = std::fs::read_to_string(&path) {
        for line in content.lines() {
            if let Ok(record) = serde_json::from_str::<TaskRecord>(line) {
                history.push(record);
            }
        }
    }
    history
}