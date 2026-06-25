use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ActionType {
    Read,
    Write,
    Execute,
    Network,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Action {
    pub command: String,
    pub args: Vec<String>,
    pub description: String,
    pub action_type: ActionType,
}

impl Action {
    pub fn new(command: &str, args: Vec<&str>, description: &str, action_type: ActionType) -> Self {
        Action {
            command: command.to_string(),
            args: args.iter().map(|s| s.to_string()).collect(),
            description: description.to_string(),
            action_type,
        }
    }

    pub fn full_command(&self) -> String {
        if self.args.is_empty() {
            self.command.clone()
        } else {
            format!("{} {}", self.command, self.args.join(" "))
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Plan {
    pub description: String,
    pub actions: Vec<Action>,
    /// How well this plan matches the goal's intent (0–1). Set by the planner;
    /// blended with the net's prediction at selection time so the goal-appropriate
    /// plan isn't buried by near-ties in the net's output.
    #[serde(default = "default_relevance")]
    pub relevance: f64,
}

fn default_relevance() -> f64 {
    0.5
}

impl Plan {
    pub fn new(description: &str, actions: Vec<Action>) -> Self {
        Plan {
            description: description.to_string(),
            actions,
            relevance: default_relevance(),
        }
    }

    pub fn step_count(&self) -> usize {
        self.actions.len()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaskRecord {
    pub goal: String,
    pub plan_description: String,
    pub step_count: usize,
    pub feature_vector: Vec<f64>,
    pub predicted_reward: f64,
    pub actual_outcome: f64,
    pub steps_completed: usize,
    pub timestamp: String,
}