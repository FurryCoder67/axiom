use crate::types::{Action, ActionType, Plan};

/// Detect conversational / question intents and return a single-plan response.
///
/// Returns `None` for task-style goals, which then fall through to the rule-based
/// planner. Two kinds of response are produced:
///   - a spoken `Answer` action (greetings, identity, capabilities, ...), and
///   - a computed answer: a normal command whose output *is* the answer
///     (e.g. "what time is it" -> `date`).
pub fn respond(goal: &str) -> Option<Plan> {
    let raw = goal.trim().to_lowercase();
    let words: Vec<String> = raw
        .split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_string()
        })
        .filter(|w| !w.is_empty())
        .collect();

    let first = words.first().map(|s| s.as_str()).unwrap_or("");
    let has = |w: &str| words.iter().any(|x| x == w);
    let phrase = |p: &str| raw.contains(p);

    // ── Identity ──
    if phrase("your name")
        || phrase("who are you")
        || phrase("what are you")
        || phrase("who r u")
        || phrase("who's this")
    {
        return Some(answer_plan(
            "Identity",
            "I'm Axiom — a terminal agent with a from-scratch neural network. I don't use a \
             large language model. I match your goal to shell-command plans, score them with my \
             net, and run the best one.",
        ));
    }

    // ── Capabilities ──
    if phrase("what can you do")
        || phrase("what do you do")
        || phrase("how do you work")
        || phrase("capabilities")
        || phrase("what are you capable")
    {
        return Some(answer_plan(
            "Capabilities",
            "I can act on goals like: list files, count files, search for text, create a file, \
             check disk usage, or show system info. I rank candidate plans with my neural net and \
             run the top one. Try: \"search for fn\" or \"how many files are here\".",
        ));
    }

    // ── Wellbeing ──
    if phrase("how are you") || phrase("how's it going") || phrase("how are u") || phrase("how do you feel") {
        return Some(answer_plan(
            "Wellbeing",
            "Running fine — weights loaded and ready. What would you like me to do?",
        ));
    }

    // ── Thanks (first word, to avoid hijacking task goals) ──
    if first == "thanks" || first == "thank" || first == "thx" || first == "ty" || phrase("thank you") {
        return Some(answer_plan("Thanks", "You're welcome!"));
    }

    // ── Goodbye (first word) ──
    if first == "bye" || first == "goodbye" || first == "cya" || phrase("see you") {
        return Some(answer_plan(
            "Goodbye",
            "Goodbye! Type 'quit' to save state and exit.",
        ));
    }

    // ── Greeting (first word, so "find hi.txt" stays a search) ──
    if first == "hi"
        || first == "hello"
        || first == "hey"
        || first == "yo"
        || first == "sup"
        || first == "howdy"
        || first == "hiya"
        || phrase("good morning")
        || phrase("good afternoon")
        || phrase("good evening")
    {
        return Some(answer_plan(
            "Greeting",
            "Hello! I'm Axiom. Give me a goal — like \"list files\" or \"how many files are here\" \
             — or ask what I can do.",
        ));
    }

    // ── Computed questions: run a command whose output is the answer ──
    if phrase("what time")
        || phrase("what's the time")
        || phrase("what is the time")
        || phrase("current time")
        || phrase("what day")
        || phrase("what's the date")
        || phrase("what is the date")
    {
        return Some(command_answer(
            "What time/date is it",
            "date",
            vec![],
            "Show the current date and time",
        ));
    }
    if phrase("who am i") || phrase("my username") || has("whoami") {
        return Some(command_answer("Who am I", "whoami", vec![], "Show the current user"));
    }
    if phrase("where am i") || phrase("what directory") || phrase("which directory") || raw == "pwd" {
        return Some(command_answer(
            "Where am I",
            "pwd",
            vec![],
            "Show the current working directory",
        ));
    }

    None
}

/// A spoken answer: the text lives in the action's `command` field and is printed
/// (never executed) by the executor.
fn answer_plan(label: &str, text: &str) -> Plan {
    let mut plan = Plan::new(label, vec![Action::new(text, vec![], label, ActionType::Answer)]);
    plan.relevance = 1.0;
    plan
}

/// A computed answer: a normal read-only command whose output answers the question.
fn command_answer(desc: &str, command: &str, args: Vec<&str>, action_desc: &str) -> Plan {
    let mut plan = Plan::new(desc, vec![Action::new(command, args, action_desc, ActionType::Read)]);
    plan.relevance = 1.0;
    plan
}
