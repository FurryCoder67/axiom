use crate::types::{Action, ActionType, Plan, TaskRecord};

/// Number of features in the plan encoding vector.
pub const FEATURE_SIZE: usize = 8;

/// Generate N candidate plans for a given goal using a rule-based planner.
/// The neural net scores them — planning stays transparent, learning focuses on reward prediction.
pub fn generate_plans(goal: &str, candidate_count: usize) -> Vec<Plan> {
    let goal_lower = goal.to_lowercase();
    let mut plans: Vec<Plan> = Vec::new();

    let wants_list = goal_lower.contains("list")
        || goal_lower.contains("show")
        || goal_lower.contains("ls")
        || goal_lower.contains("display");
    let wants_create = goal_lower.contains("create")
        || goal_lower.contains("make")
        || goal_lower.contains("new")
        || goal_lower.contains("touch");
    let wants_search = goal_lower.contains("search")
        || goal_lower.contains("find")
        || goal_lower.contains("grep")
        || goal_lower.contains("look");
    let wants_count = goal_lower.contains("count")
        || goal_lower.contains("how many")
        || goal_lower.contains("number of");
    let wants_clean = goal_lower.contains("clean")
        || goal_lower.contains("remove")
        || goal_lower.contains("delete")
        || goal_lower.contains("rm ");
    let wants_info = goal_lower.contains("info")
        || goal_lower.contains("status")
        || goal_lower.contains("check")
        || goal_lower.contains("inspect");
    let wants_disk = goal_lower.contains("disk")
        || goal_lower.contains("space")
        || goal_lower.contains("memory")
        || goal_lower.contains("usage");

    // ── Plan 1: Minimal / direct approach ──
    if wants_list {
        plans.push(Plan::new(
            "List directory contents",
            vec![Action::new(
                "ls",
                vec!["-la"],
                "List all files in current directory",
                ActionType::Read,
            )],
        ));
    } else if wants_create {
        let filename = extract_target(goal).unwrap_or_else(|| "new_file.txt".to_string());
        plans.push(Plan::new(
            &format!("Create file: {}", filename),
            vec![Action::new(
                "touch",
                vec![&filename],
                &format!("Create empty file {}", filename),
                ActionType::Write,
            )],
        ));
    } else if wants_search {
        let pattern = extract_target(goal).unwrap_or_else(|| "pattern".to_string());
        plans.push(Plan::new(
            &format!("Search for '{}'", pattern),
            vec![Action::new(
                "grep",
                vec!["-r", &pattern, "."],
                &format!("Recursively search for '{}'", pattern),
                ActionType::Read,
            )],
        ));
    } else if wants_count {
        plans.push(Plan::new(
            "Count files in directory",
            vec![Action::new(
                "sh",
                vec!["-c", "ls -1A | wc -l"],
                "Count entries in the current directory",
                ActionType::Execute,
            )],
        ));
    } else if wants_disk {
        plans.push(Plan::new(
            "Check disk usage",
            vec![Action::new(
                "df",
                vec!["-h"],
                "Show disk space usage in human-readable format",
                ActionType::Read,
            )],
        ));
    } else if wants_clean {
        let target = extract_target(goal).unwrap_or_else(|| "temp".to_string());
        plans.push(Plan::new(
            &format!("Remove {}", target),
            vec![Action::new(
                "rm",
                vec![&target],
                &format!("Remove {}", target),
                ActionType::Write,
            )],
        ));
    } else if wants_info {
        plans.push(Plan::new(
            "Show system information",
            vec![Action::new(
                "uname",
                vec!["-a"],
                "Print system information",
                ActionType::Read,
            )],
        ));
    } else {
        plans.push(Plan::new(
            "Explore current directory",
            vec![Action::new(
                "ls",
                vec!["-la"],
                "List all files in current directory",
                ActionType::Read,
            )],
        ));
    }

    // ── Plan 2: Two-step approach (recon + execute) ──
    if wants_search {
        let pattern = extract_target(goal).unwrap_or_else(|| "pattern".to_string());
        plans.push(Plan::new(
            &format!("Survey then search for '{}'", pattern),
            vec![
                Action::new("ls", vec!["-la"], "Survey directory structure", ActionType::Read),
                Action::new(
                    "grep",
                    vec!["-rn", &pattern, "."],
                    &format!("Search for '{}' with line numbers", pattern),
                    ActionType::Read,
                ),
            ],
        ));
    } else if wants_list {
        plans.push(Plan::new(
            "List with disk usage summary",
            vec![
                Action::new("ls", vec!["-la"], "List all files", ActionType::Read),
                Action::new("du", vec!["-sh", "."], "Show directory size", ActionType::Read),
            ],
        ));
    } else if wants_create {
        let filename = extract_target(goal).unwrap_or_else(|| "new_file.txt".to_string());
        plans.push(Plan::new(
            &format!("Verify then create: {}", filename),
            vec![
                Action::new("pwd", vec![], "Show current directory", ActionType::Read),
                Action::new(
                    "touch",
                    vec![&filename],
                    &format!("Create file {}", filename),
                    ActionType::Write,
                ),
            ],
        ));
    } else {
        plans.push(Plan::new(
            "Gather context before acting",
            vec![
                Action::new("pwd", vec![], "Show current directory", ActionType::Read),
                Action::new("ls", vec!["-la"], "List directory contents", ActionType::Read),
            ],
        ));
    }

    // ── Plan 3: Thorough approach (more context) ──
    plans.push(Plan::new(
        "Thorough exploration before action",
        vec![
            Action::new("pwd", vec![], "Show current directory", ActionType::Read),
            Action::new("ls", vec!["-la"], "List all files", ActionType::Read),
            Action::new("file", vec!["*"], "Identify file types", ActionType::Read),
        ],
    ));

    // ── Plan 4: Alternative approach (different commands) ──
    if wants_count {
        plans.push(Plan::new(
            "Alternative: count via find",
            vec![Action::new(
                "sh",
                vec!["-c", "find . -maxdepth 1 -mindepth 1 | wc -l"],
                "Count files using find",
                ActionType::Execute,
            )],
        ));
    } else if wants_list {
        plans.push(Plan::new(
            "Alternative: use find to list files",
            vec![Action::new(
                "find",
                vec![".", "-maxdepth", "1", "-print"],
                "Find files in current directory",
                ActionType::Read,
            )],
        ));
    } else if wants_search {
        let pattern = extract_target(goal).unwrap_or_else(|| "pattern".to_string());
        plans.push(Plan::new(
            &format!("Alternative: use find for '{}'", pattern),
            vec![Action::new(
                "find",
                vec![".", "-name", &format!("*{}*", pattern)],
                &format!("Find files matching '{}'", pattern),
                ActionType::Read,
            )],
        ));
    } else if wants_create {
        let filename = extract_target(goal).unwrap_or_else(|| "new_file.txt".to_string());
        plans.push(Plan::new(
            &format!("Alternative: create with echo: {}", filename),
            vec![Action::new(
                "sh",
                vec!["-c", &format!("echo '' > {}", filename)],
                &format!("Create empty file {} via shell", filename),
                ActionType::Execute,
            )],
        ));
    } else {
        plans.push(Plan::new(
            "Alternative: check environment",
            vec![
                Action::new("whoami", vec![], "Show current user", ActionType::Read),
                Action::new("date", vec![], "Show current date", ActionType::Read),
            ],
        ));
    }

    // Tag each plan with a goal-relevance weight by slot. Plans are generated in a
    // fixed semantic order: 0 = direct/intent-matched, 1 = recon-then-act,
    // 2 = thorough exploration, 3 = alternative tool. The direct plan most closely
    // matches the goal's intent, so it gets the highest relevance.
    const RELEVANCE_BY_SLOT: [f64; 4] = [1.0, 0.7, 0.5, 0.65];
    for (i, plan) in plans.iter_mut().enumerate() {
        plan.relevance = RELEVANCE_BY_SLOT.get(i).copied().unwrap_or(0.5);
    }

    // Trim or pad to candidate_count
    if plans.len() > candidate_count {
        plans.truncate(candidate_count);
    }
    while plans.len() < candidate_count {
        let last = plans.last().cloned().unwrap_or_else(|| {
            Plan::new(
                "List directory",
                vec![Action::new("ls", vec!["-la"], "List files", ActionType::Read)],
            )
        });
        plans.push(last);
    }

    plans
}

/// Naive target extraction: looks for a word after common prepositions.
fn extract_target(goal: &str) -> Option<String> {
    let words: Vec<&str> = goal.split_whitespace().collect();
    for (i, word) in words.iter().enumerate() {
        let w = word.to_lowercase();
        if (w == "for"
            || w == "named"
            || w == "called"
            || w == "file"
            || w == "create"
            || w == "find"
            || w == "called"
            || w == "remove"
            || w == "delete"
            || w == "search")
            && i + 1 < words.len()
        {
            return Some(
                words[i + 1]
                    .trim_matches(|c: char| !c.is_alphanumeric() && c != '.' && c != '_' && c != '-')
                    .to_string(),
            );
        }
    }
    // Fallback: last word
    words.last().map(|w| {
        w.trim_matches(|c: char| !c.is_alphanumeric() && c != '.' && c != '_' && c != '-')
            .to_string()
    })
}

/// Encode a candidate plan as a fixed-length feature vector for the neural net.
///
/// Features:
///   0: plan length (normalized)
///   1: proportion of Read actions
///   2: proportion of Write actions
///   3: proportion of Execute actions
///   4: proportion of Network actions
///   5: risk level (weighted by action types)
///   6: goal keyword overlap with plan text
///   7: historical success rate for similar plans
pub fn encode_plan(plan: &Plan, goal: &str, history: &[TaskRecord]) -> Vec<f64> {
    let mut features = vec![0.0; FEATURE_SIZE];

    let total = plan.actions.len() as f64;

    // 0: Plan length
    features[0] = total / 10.0;

    // 1-4: Command type proportions
    let mut reads = 0.0;
    let mut writes = 0.0;
    let mut executes = 0.0;
    let mut networks = 0.0;
    for action in &plan.actions {
        match action.action_type {
            ActionType::Read => reads += 1.0,
            ActionType::Write => writes += 1.0,
            ActionType::Execute => executes += 1.0,
            ActionType::Network => networks += 1.0,
            // A spoken answer is informational and risk-free; treat like a read.
            ActionType::Answer => reads += 1.0,
        }
    }
    features[1] = reads / total;
    features[2] = writes / total;
    features[3] = executes / total;
    features[4] = networks / total;

    // 5: Risk level (writes and executes are riskier)
    features[5] = (writes * 0.5 + executes * 0.7 + networks * 0.3) / total;

    // 6: Goal keyword overlap
    let plan_text = format!(
        "{} {}",
        plan.description,
        plan.actions
            .iter()
            .map(|a| a.full_command())
            .collect::<Vec<_>>()
            .join(" ")
    )
    .to_lowercase();
    let goal_lower = goal.to_lowercase();
    let goal_words: Vec<&str> = goal_lower.split_whitespace().collect();
    let mut overlap = 0.0;
    for word in &goal_words {
        if word.len() > 2 && plan_text.contains(word) {
            overlap += 1.0;
        }
    }
    features[6] = overlap / goal_words.len().max(1) as f64;

    // 7: Historical success rate for similar plans
    if history.is_empty() {
        features[7] = 0.5; // neutral prior
    } else {
        let similar: Vec<&TaskRecord> = history
            .iter()
            .filter(|r| (r.step_count as i32 - plan.step_count() as i32).abs() <= 2)
            .collect();
        if similar.is_empty() {
            features[7] = 0.5;
        } else {
            let successes: f64 = similar.iter().map(|r| r.actual_outcome).sum();
            features[7] = successes / similar.len() as f64;
        }
    }

    features
}