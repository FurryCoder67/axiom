use crate::types::{Action, ActionType, Plan};
use serde::{Deserialize, Serialize};

/// Conversation memory. `user_name` and `turns` persist across sessions
/// (`~/.axiom/conversation.json`); the rest is session-scoped dialogue state.
#[derive(Default, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub user_name: Option<String>,
    pub turns: u64,
    #[serde(skip)]
    pub last_intent: Option<String>,
    /// Set when Axiom asks something and expects a contextual reply next turn,
    /// e.g. "wellbeing" after "how are you?" so it can read your answer.
    #[serde(skip)]
    pub awaiting: Option<String>,
}

/// Handle conversational input, updating memory in `convo`. Returns `None` for
/// task-style goals, which fall through to the rule-based planner.
pub fn respond(goal: &str, convo: &mut Conversation) -> Option<Plan> {
    let (plan, intent) = classify(goal, convo)?;
    convo.turns += 1;
    convo.last_intent = Some(intent);
    Some(plan)
}

fn classify(goal: &str, convo: &mut Conversation) -> Option<(Plan, String)> {
    let raw = goal.trim().to_lowercase();
    let words: Vec<String> = raw
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|w| !w.is_empty())
        .collect();
    let first = words.first().map(|s| s.as_str()).unwrap_or("");
    let has = |w: &str| words.iter().any(|x| x == w);
    let phrase = |p: &str| raw.contains(p);
    let name = convo.user_name.clone();

    // ── Contextual follow-up: we just asked how they're doing ──
    let awaiting = convo.awaiting.take();
    if awaiting.as_deref() == Some("wellbeing") {
        if let Some(positive) = feeling_sentiment(&raw) {
            let reply = if positive {
                "Glad to hear it! What can I do for you?"
            } else {
                "Sorry to hear that. Maybe I can help — give me a goal and I'll get to work."
            };
            return Some((answer(reply, &name), "wellbeing_reply".into()));
        }
        // Not a feeling reply — fall through to normal classification.
    }

    // ── Name capture: "my name is X", "i'm X", "call me X" ──
    if let Some(n) = capture_name(&raw, &words) {
        convo.user_name = Some(n.clone());
        let text = format!("Nice to meet you, {}! How can I help?", n);
        return Some((answer_text(text, &None), "name_set".into()));
    }

    // ── Ask for remembered name ──
    if phrase("my name") || phrase("do you know who i am") || phrase("remember me") {
        let text = match &name {
            Some(n) => format!("Of course — you're {}.", n),
            None => "I don't know your name yet. Tell me with \"my name is ...\".".to_string(),
        };
        return Some((answer_text(text, &None), "name_get".into()));
    }

    // ── Arithmetic: "what is 2 + 2", "3 times 4", "calculate 10 / 4" ──
    if let Some(result) = try_math(&raw) {
        return Some((answer_text(result, &name), "math".into()));
    }

    // ── Greeting (first word, so "find hi.txt" stays a search) ──
    if matches!(first, "hi" | "hello" | "hey" | "yo" | "sup" | "howdy" | "hiya")
        || phrase("good morning")
        || phrase("good afternoon")
        || phrase("good evening")
    {
        let text = match &name {
            Some(n) if convo.turns > 0 => format!("Hello again, {}! What shall we do?", n),
            Some(n) => format!("Hello, {}!", n),
            None => pick(
                convo.turns,
                &[
                    "Hello! I'm Axiom. Give me a goal, ask what I can do, or just chat.",
                    "Hey there! Tell me a goal like \"list files\", or ask me something.",
                    "Hi! I'm Axiom — what can I do for you?",
                ],
            )
            .to_string(),
        };
        return Some((answer_text(text, &None), "greeting".into()));
    }

    // ── Wellbeing (asks back, then expects a reply) ──
    if phrase("how are you") || phrase("how's it going") || phrase("how are u") || phrase("how do you feel") {
        convo.awaiting = Some("wellbeing".into());
        return Some((
            answer("Running fine — weights loaded and ready. How about you?", &name),
            "wellbeing".into(),
        ));
    }

    // ── User volunteers their mood ──
    if matches!(first, "im" | "i'm" | "i") || has("feeling") || has("feel") {
        if let Some(positive) = feeling_sentiment(&raw) {
            let reply = if positive {
                "Nice! Glad things are good. What would you like to do?"
            } else {
                "Sorry to hear that. Want me to take something off your plate? Give me a goal."
            };
            return Some((answer(reply, &name), "mood".into()));
        }
    }

    // ── Identity ──
    if phrase("your name")
        || phrase("who are you")
        || phrase("what are you")
        || phrase("who r u")
        || phrase("what's your name")
    {
        return Some((
            answer(
                "I'm Axiom — a terminal agent with a from-scratch neural network, no large \
                 language model. I match your goal to shell-command plans, score them with my net, \
                 and run the best one.",
                &name,
            ),
            "identity".into(),
        ));
    }

    // ── Creator / origin ──
    if phrase("who made you") || phrase("who created you") || phrase("who built you") || phrase("where do you come from") {
        return Some((
            answer(
                "I'm written in pure Rust — a from-scratch feedforward neural net with backprop, \
                 no ML frameworks or pretrained models.",
                &name,
            ),
            "origin".into(),
        ));
    }

    // ── Age ──
    if phrase("how old") || phrase("your age") {
        return Some((
            answer("I'm just a program — no age, only a weights file that gets smarter each run.", &name),
            "age".into(),
        ));
    }

    // ── Capabilities / help ──
    if phrase("what can you do")
        || phrase("what do you do")
        || phrase("how do you work")
        || phrase("capabilities")
        || phrase("help me")
    {
        return Some((
            answer(
                "I can: list files, count files, search for text, create a file, check disk usage, \
                 or show system info — and chat a bit. Try \"search for fn\" or \"how many files \
                 are here\". Type 'help' for REPL commands.",
                &name,
            ),
            "capabilities".into(),
        ));
    }

    // ── What are you doing / what's up ──
    if phrase("what are you doing") || phrase("whats up") || phrase("what's up") || phrase("wassup") {
        return Some((
            answer("Just waiting for a goal to plan and run. What's on your mind?", &name),
            "status_chat".into(),
        ));
    }

    // ── Jokes ──
    if phrase("joke") || phrase("make me laugh") || phrase("something funny") {
        let joke = pick(
            convo.turns,
            &[
                "Why do programmers prefer dark mode? Because light attracts bugs.",
                "There are 10 kinds of people: those who understand binary and those who don't.",
                "I'd tell you a UDP joke, but you might not get it.",
                "A SQL query walks into a bar, sees two tables, and asks: \"Can I join you?\"",
            ],
        );
        return Some((answer(joke, &name), "joke".into()));
    }

    // ── Thanks ──
    if first == "thanks" || first == "thank" || first == "thx" || first == "ty" || phrase("thank you") {
        return Some((
            answer(pick(convo.turns, &["You're welcome!", "Anytime!", "Happy to help."]), &name),
            "thanks".into(),
        ));
    }

    // ── Goodbye ──
    if matches!(first, "bye" | "goodbye" | "cya") || phrase("see you") || phrase("good night") {
        return Some((
            answer("Goodbye! Type 'quit' to save state and exit.", &name),
            "goodbye".into(),
        ));
    }

    // ── Computed questions (a real command's output is the answer) ──
    if phrase("what time")
        || phrase("what's the time")
        || phrase("what is the time")
        || phrase("current time")
        || phrase("what day")
        || phrase("what's the date")
        || phrase("what is the date")
    {
        return Some((command_answer("What time/date is it", "date", vec![], "Show the current date and time"), "time".into()));
    }
    if phrase("who am i") || phrase("my username") || has("whoami") {
        return Some((command_answer("Who am I", "whoami", vec![], "Show the current user"), "whoami".into()));
    }
    if (phrase("where am i") || phrase("what directory") || phrase("which directory") || raw == "pwd")
        && !has("file") && !has("files")
    {
        return Some((command_answer("Where am I", "pwd", vec![], "Show the current working directory"), "pwd".into()));
    }

    // ── Graceful catch-all: clearly a question, but not a task we plan for ──
    if (raw.ends_with('?') || is_conversational_opener(first)) && !looks_like_task(&words) {
        return Some((
            answer(
                "I'm a simple rule-based agent, so I can't answer open-ended questions like an \
                 LLM. But I can run real commands — try \"list files\", \"search for <text>\", or \
                 ask \"what can you do\".",
                &name,
            ),
            "fallback".into(),
        ));
    }

    None
}

/// Detect "my name is X" / "i'm X" / "call me X" and return a clean name.
fn capture_name(raw: &str, words: &[String]) -> Option<String> {
    let anchors = [("my name is", 3), ("i am", 2), ("i'm", 1), ("im", 1), ("call me", 2)];
    for (anchor, skip) in anchors {
        if raw.starts_with(anchor) || raw.contains(&format!(" {} ", anchor)) {
            // Take the token right after the anchor.
            let after: Vec<&String> = words.iter().skip_while(|w| {
                !anchor.split_whitespace().last().map(|a| *w == a).unwrap_or(false)
            }).skip(1).collect();
            if let Some(cand) = after.first() {
                let _ = skip;
                let lc = cand.to_lowercase();
                // Don't mistake feelings/stopwords for a name.
                if feeling_word(&lc) || matches!(lc.as_str(), "not" | "here" | "sorry" | "going" | "a" | "the" | "your") {
                    return None;
                }
                if cand.chars().all(|c| c.is_alphabetic()) && (2..=20).contains(&cand.len()) {
                    let mut chars = cand.chars();
                    let name: String = chars.next().unwrap().to_uppercase().collect::<String>()
                        + &chars.as_str().to_lowercase();
                    return Some(name);
                }
            }
        }
    }
    None
}

/// Classify a short "how are you" reply: Some(true)=positive, Some(false)=negative.
fn feeling_sentiment(raw: &str) -> Option<bool> {
    let pos = ["good", "great", "fine", "ok", "okay", "well", "happy", "awesome", "fantastic", "amazing", "alright", "not bad"];
    let neg = ["bad", "sad", "tired", "terrible", "awful", "meh", "down", "stressed", "not good", "not great", "rough", "exhausted"];
    if neg.iter().any(|w| raw.contains(w)) {
        return Some(false);
    }
    if pos.iter().any(|w| raw.contains(w)) {
        return Some(true);
    }
    None
}

fn feeling_word(w: &str) -> bool {
    matches!(
        w,
        "good" | "great" | "fine" | "ok" | "okay" | "well" | "happy" | "awesome" | "fantastic"
            | "amazing" | "alright" | "bad" | "sad" | "tired" | "terrible" | "awful" | "meh"
            | "down" | "stressed" | "rough" | "exhausted"
    )
}

/// Best-effort two-operand arithmetic: "what is 2 + 2", "3 times 4", "10 / 4".
fn try_math(raw: &str) -> Option<String> {
    let mut s = raw.to_string();
    for p in ["what is", "what's", "whats", "calculate", "compute", "how much is", "how many is"] {
        s = s.replace(p, " ");
    }
    s = s.replace(" plus ", " + ")
        .replace(" minus ", " - ")
        .replace(" times ", " * ")
        .replace(" x ", " * ")
        .replace(" multiplied by ", " * ")
        .replace(" divided by ", " / ")
        .replace(" over ", " / ");

    let mut nums: Vec<f64> = Vec::new();
    let mut op: Option<char> = None;
    for tok in s.split_whitespace() {
        if let Ok(n) = tok.parse::<f64>() {
            nums.push(n);
        } else if tok.len() == 1 {
            let c = tok.chars().next().unwrap();
            if matches!(c, '+' | '-' | '*' | '/') {
                op = Some(c);
            }
        }
    }
    if nums.len() != 2 {
        return None;
    }
    let op = op?;
    let (a, b) = (nums[0], nums[1]);
    let r = match op {
        '+' => a + b,
        '-' => a - b,
        '*' => a * b,
        '/' => {
            if b == 0.0 {
                return Some("That's undefined — can't divide by zero.".to_string());
            }
            a / b
        }
        _ => return None,
    };
    let pretty = if r.fract() == 0.0 {
        format!("{}", r as i64)
    } else {
        format!("{:.4}", r).trim_end_matches('0').trim_end_matches('.').to_string()
    };
    Some(format!("{} {} {} = {}", trim_num(a), op, trim_num(b), pretty))
}

fn trim_num(n: f64) -> String {
    if n.fract() == 0.0 { format!("{}", n as i64) } else { format!("{}", n) }
}

fn is_conversational_opener(first: &str) -> bool {
    matches!(
        first,
        "do" | "are" | "can" | "will" | "why" | "what" | "who" | "how" | "tell" | "give" | "is" | "does" | "would"
    )
}

/// Mirror the planner's task keywords so the catch-all never hijacks a real goal.
fn looks_like_task(words: &[String]) -> bool {
    const KW: &[&str] = &[
        "list", "show", "ls", "display", "create", "make", "new", "touch", "search", "find",
        "grep", "look", "count", "clean", "remove", "delete", "rm", "info", "status", "check",
        "inspect", "disk", "space", "memory", "usage", "file", "files", "directory",
    ];
    words.iter().any(|w| KW.contains(&w.as_str()))
}

/// Deterministic variety: rotate response options by turn count.
fn pick<'a>(turns: u64, options: &[&'a str]) -> &'a str {
    options[(turns as usize) % options.len()]
}

/// A spoken answer, optionally prefixed with the user's name for a personal touch.
fn answer(text: &str, name: &Option<String>) -> Plan {
    answer_text(text.to_string(), name)
}

fn answer_text(text: String, name: &Option<String>) -> Plan {
    let final_text = match name {
        Some(_n) => text, // keep replies clean; name already used where it reads naturally
        None => text,
    };
    let mut plan = Plan::new("Chat", vec![Action::new(&final_text, vec![], "Conversational reply", ActionType::Answer)]);
    plan.relevance = 1.0;
    plan
}

/// A computed answer: a real read-only command whose output answers the question.
fn command_answer(desc: &str, command: &str, args: Vec<&str>, action_desc: &str) -> Plan {
    let mut plan = Plan::new(desc, vec![Action::new(command, args, action_desc, ActionType::Read)]);
    plan.relevance = 1.0;
    plan
}
