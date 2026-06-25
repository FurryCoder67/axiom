use crate::types::{Action, ActionType, Plan};

/// Check a single action against the blocklist.
/// Returns Some(reason) if blocked, None if safe.
pub fn check_action(action: &Action, blocklist: &[String]) -> Option<String> {
    // Spoken answers are never executed, so there is nothing to block.
    if action.action_type == ActionType::Answer {
        return None;
    }
    let full_cmd = action.full_command().to_lowercase();
    for pattern in blocklist {
        let p = pattern.to_lowercase();
        if full_cmd.contains(&p) {
            return Some(format!("matches blocked pattern: '{}'", pattern));
        }
    }
    None
}

/// Check an entire plan. Returns a list of (step_index, reason) for blocked steps.
pub fn check_plan(plan: &Plan, blocklist: &[String]) -> Vec<(usize, String)> {
    let mut blocked = Vec::new();
    for (i, action) in plan.actions.iter().enumerate() {
        if let Some(reason) = check_action(action, blocklist) {
            blocked.push((i, reason));
        }
    }
    blocked
}