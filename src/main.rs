mod neural_net;
mod planner;
mod safety;
mod storage;
mod terminal;
mod types;

use neural_net::NeuralNet;
use planner::FEATURE_SIZE;
use serde::Deserialize;
use terminal::*;
use types::{Plan, TaskRecord};

use std::io;

// ── Config ───────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct Config {
    pub net: NetConfig,
    pub agent: AgentConfig,
    pub safety: SafetyConfig,
}

#[derive(Deserialize)]
pub struct NetConfig {
    pub layer_sizes: Vec<usize>,
    pub learning_rate: f64,
}

#[derive(Deserialize)]
pub struct AgentConfig {
    pub plan_candidate_count: usize,
    pub data_dir: String,
}

#[derive(Deserialize)]
pub struct SafetyConfig {
    pub blocklist: Vec<String>,
}

fn load_config() -> Config {
    if let Ok(content) = std::fs::read_to_string("config.toml") {
        if let Ok(config) = toml::from_str::<Config>(&content) {
            return config;
        }
    }
    // Built-in defaults
    Config {
        net: NetConfig {
            layer_sizes: vec![FEATURE_SIZE, 16, 8, 1],
            learning_rate: 0.1,
        },
        agent: AgentConfig {
            plan_candidate_count: 4,
            data_dir: "~/.axiom".to_string(),
        },
        safety: SafetyConfig {
            blocklist: vec![
                "rm -rf /".to_string(),
                "sudo".to_string(),
                "mkfs".to_string(),
                "dd if=".to_string(),
                ":(){:|:&};:".to_string(),
                "chmod 777 /".to_string(),
                "shutdown".to_string(),
                "reboot".to_string(),
                "> /dev/sda".to_string(),
            ],
        },
    }
}

// ── Agent ────────────────────────────────────────────────────────────

struct Axiom {
    config: Config,
    net: NeuralNet,
    history: Vec<TaskRecord>,
    interrupt_flag: bool,
    current_goal: Option<String>,
}

impl Axiom {
    fn new() -> Self {
        let config = load_config();

        let net = storage::load_weights(&config).unwrap_or_else(|| {
            NeuralNet::new(&config.net.layer_sizes, config.net.learning_rate)
        });

        let history = storage::load_history(&config);

        Axiom {
            config,
            net,
            history,
            interrupt_flag: false,
            current_goal: None,
        }
    }

    fn print_banner(&self) {
        println!();
        println!("{}", bold("  ╔══════════════════════════════════════╗"));
        println!("{}", bold("  ║              A X I O M                 ║"));
        println!("{}", bold("  ╚══════════════════════════════════════╝"));
        println!();
        println!("  {}Terminal AI agent — from-scratch neural net{}", DIM, RESET);
        println!(
            "  {}Net: {} | Params: {} | History: {} tasks | Data: {}{}",
            dim("→"),
            format_layer_sizes(&self.net),
            self.net.param_count(),
            self.history.len(),
            self.config.agent.data_dir,
            RESET
        );
        println!(
            "  {}Type a goal, or 'help' for commands.{}\n",
            dim(""), RESET
        );
    }

    fn run(&mut self) {
        self.print_banner();

        let stdin = io::stdin();
        loop {
            print_prompt();
            let mut input = String::new();
            if stdin.read_line(&mut input).is_err() {
                break;
            }
            let trimmed = input.trim();
            if trimmed.is_empty() {
                continue;
            }

            if trimmed == "quit" || trimmed == "exit" {
                println!("{}Saving state and exiting...{}", dim(""), RESET);
                storage::save_weights(&self.net, &self.config);
                storage::save_history(&self.history, &self.config);
                break;
            }

            match trimmed {
                "help" => self.show_help(),
                "status" => self.show_status(),
                "interrupt" | "stop" => self.handle_interrupt(),
                "history" => self.show_history(),
                "inspect weights" | "weights" => self.inspect_weights(),
                cmd if cmd.starts_with("redirect ") => {
                    let goal = &cmd["redirect ".len()..];
                    self.redirect(goal);
                    storage::save_weights(&self.net, &self.config);
                    storage::save_history(&self.history, &self.config);
                }
                goal => {
                    self.handle_goal(goal);
                    storage::save_weights(&self.net, &self.config);
                    storage::save_history(&self.history, &self.config);
                }
            }
        }
    }

    fn show_help(&self) {
        println!("  {}Commands:{}", bold(""), RESET);
        println!("    {}<any text>{}       — give Axiom a goal to plan and execute", dim(""), RESET);
        println!("    {}status{}           — show current goal and net state", dim(""), RESET);
        println!("    {}interrupt{}        — halt the current task", dim(""), RESET);
        println!("    {}redirect <goal>{} — abandon current task, pivot to a new goal", dim(""), RESET);
        println!("    {}inspect weights{} — show neural net layer shapes and weight stats", dim(""), RESET);
        println!("    {}history{}         — show past tasks and their outcomes", dim(""), RESET);
        println!("    {}help{}            — show this message", dim(""), RESET);
        println!("    {}quit{}            — save and exit", dim(""), RESET);
        println!();
    }

    fn show_status(&self) {
        println!("  {}Status{}", bold(""), RESET);
        match &self.current_goal {
            Some(goal) => println!("    Current goal: {}", yellow(goal)),
            None => println!("    No active goal"),
        }
        println!("    Tasks completed: {}", self.history.len());
        println!("    Net architecture: {}", format_layer_sizes(&self.net));
        println!("    Total parameters: {}", self.net.param_count());
        let success_rate = if self.history.is_empty() {
            0.0
        } else {
            self.history.iter().filter(|r| r.actual_outcome >= 1.0).count() as f64
                / self.history.len() as f64
        };
        println!("    Success rate: {:.1}%", success_rate * 100.0);
        println!();
    }

    fn handle_interrupt(&mut self) {
        self.interrupt_flag = true;
        self.current_goal = None;
        println!("{}", red("Interrupt flag set — will halt at next step."));
        println!();
    }

    fn redirect(&mut self, goal: &str) {
        self.interrupt_flag = true;
        self.current_goal = Some(goal.to_string());
        println!("{}Redirecting to: {}{}", yellow(""), goal, RESET);
        self.handle_goal(goal);
    }

    fn show_history(&self) {
        if self.history.is_empty() {
            println!("  {}No task history yet.{}", dim(""), RESET);
            println!();
            return;
        }
        println!("  {}Task History ({} entries){}", bold(""), self.history.len(), RESET);
        for (i, record) in self.history.iter().rev().take(20).enumerate() {
            let outcome_label = if record.actual_outcome >= 1.0 {
                green("✓ success")
            } else {
                red("✗ failed")
            };
            let goal_preview: String = record.goal.chars().take(50).collect();
            println!(
                "    [{}] {} — {} (predicted: {:.2}, steps: {}/{})",
                self.history.len() - i,
                outcome_label,
                goal_preview,
                record.predicted_reward,
                record.steps_completed,
                record.step_count
            );
        }
        println!();
    }

    fn inspect_weights(&self) {
        println!("  {}Neural Network Summary{}", bold(""), RESET);
        print!("{}", self.net.summary());
        println!("  Total parameters: {}", self.net.param_count());
        println!();
    }

    fn handle_goal(&mut self, goal: &str) {
        self.interrupt_flag = false;
        self.current_goal = Some(goal.to_string());

        println!("\n  {}Goal: {}{}\n", bold(""), yellow(goal), RESET);

        // 1. Generate candidate plans
        let plans = planner::generate_plans(goal, self.config.agent.plan_candidate_count);

        if plans.is_empty() {
            println!("{}", red("  No plans could be generated for this goal."));
            println!();
            self.current_goal = None;
            return;
        }

        // 2. Score each plan with the neural net
        let mut scored: Vec<(Plan, Vec<f64>, f64)> = plans
            .iter()
            .map(|plan| {
                let features = planner::encode_plan(plan, goal, &self.history);
                let score = self.net.predict(&features);
                (plan.clone(), features, score)
            })
            .collect();

        // 3. Sort by score descending
        scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // 4. Display plans with scores
        println!("  {}Candidate Plans:{}", bold(""), RESET);
        for (i, (plan, _, score)) in scored.iter().enumerate() {
            let marker = if i == 0 { "▶" } else { " " };
            let label_color = if i == 0 { yellow("") } else { dim("") };
            println!(
                "    {} {}Plan {}{}: {}",
                marker, label_color, i + 1, RESET, plan.description
            );
            println!("      {}", score_bar(*score));
            for (j, action) in plan.actions.iter().enumerate() {
                println!(
                    "        {}. {}{}{} — {}",
                    j + 1,
                    dim(""),
                    action.full_command(),
                    RESET,
                    action.description
                );
            }
        }
        println!();

        // 5. Execute the highest-scoring plan
        let (best_plan, best_features, best_score) = &scored[0];

        // Safety check
        let blocked = safety::check_plan(best_plan, &self.config.safety.blocklist);
        if !blocked.is_empty() {
            for (step, reason) in &blocked {
                println!("  {}Blocked step {}: {}{}", red(""), step + 1, reason, RESET);
            }
            println!("{}", red("  Plan contains blocked commands. Aborting."));
            println!();
            self.current_goal = None;
            return;
        }

        println!("  {}Executing Plan 1 (score: {:.2})...{}\n", green(""), best_score, RESET);

        let (success, steps_completed) = self.execute_plan(best_plan);

        let outcome = if success { 1.0 } else { 0.0 };

        // 6. Train the net on the outcome
        self.net.train(best_features, outcome);

        // 7. Record in history
        let record = TaskRecord {
            goal: goal.to_string(),
            plan_description: best_plan.description.clone(),
            step_count: best_plan.step_count(),
            feature_vector: best_features.clone(),
            predicted_reward: *best_score,
            actual_outcome: outcome,
            steps_completed,
            timestamp: current_timestamp(),
        };
        self.history.push(record);

        if success {
            println!(
                "\n  {}✓ Task completed successfully.{} (trained on outcome: success)",
                green(""), RESET
            );
        } else {
            println!(
                "\n  {}✗ Task failed or was interrupted.{} (trained on outcome: failure)",
                red(""), RESET
            );
        }
        println!();

        self.current_goal = None;
    }

    fn execute_plan(&mut self, plan: &Plan) -> (bool, usize) {
        for (i, action) in plan.actions.iter().enumerate() {
            if self.interrupt_flag {
                println!("  {}Interrupted at step {}.{}", red(""), i + 1, RESET);
                return (false, i);
            }

            println!(
                "  {}[step {}/{}] {}{}",
                dim(""),
                i + 1,
                plan.actions.len(),
                action.description,
                RESET
            );
            println!("  {}$ {}{}", dim(""), action.full_command(), RESET);

            let result = std::process::Command::new(&action.command)
                .args(&action.args)
                .output();

            match result {
                Ok(output) => {
                    if !output.stdout.is_empty() {
                        let stdout_str = String::from_utf8_lossy(&output.stdout);
                        for line in stdout_str.lines() {
                            println!("  {}  {}{}", dim(""), line, RESET);
                        }
                    }
                    if !output.stderr.is_empty() {
                        let stderr_str = String::from_utf8_lossy(&output.stderr);
                        for line in stderr_str.lines() {
                            println!("  {}  {}{}", red(""), line, RESET);
                        }
                    }
                    if !output.status.success() {
                        println!(
                            "  {}Command exited with code {:?}{}",
                            red(""),
                            output.status.code(),
                            RESET
                        );
                        return (false, i + 1);
                    }
                }
                Err(e) => {
                    println!("  {}Failed to execute: {}{}", red(""), e, RESET);
                    return (false, i);
                }
            }
        }
        (true, plan.actions.len())
    }
}

// ── Helpers ──────────────────────────────────────────────────────────

fn format_layer_sizes(net: &NeuralNet) -> String {
    let sizes: Vec<String> = net
        .layers
        .iter()
        .map(|l| format!("{}x{}", l.weights[0].len(), l.weights.len()))
        .collect();
    format!("[{}] (lr={})", sizes.join(" → "), net.learning_rate)
}

fn current_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{}", now)
}

// ── Entry point ─────────────────────────────────────────────────────

fn main() {
    let mut axiom = Axiom::new();
    axiom.run();
}