"""Central configuration for GapDetect. All detector thresholds live here."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DetectorConfig:
    """Tunable thresholds shared by detectors and the evidence engine."""

    # --- Unanswered questions / requests ---
    unanswered_lookahead_messages: int = 12
    unanswered_lookahead_minutes: float = 180.0
    answer_content_overlap_min: int = 1  # min shared content tokens to count as answer

    # --- Ignored / directed messages ---
    ignored_lookahead_messages: int = 10
    ignored_lookahead_minutes: float = 120.0
    ignored_min_words: int = 5  # directed msgs shorter than this are chatter
    ignored_min_window: int = 3  # need this many following messages to judge silence

    # --- Repeated clarification ---
    clarification_similarity_threshold: float = 0.45  # cosine sim between re-asks
    clarification_reask_window: int = 15  # look back this many messages for similar q
    clarification_use_embeddings: bool = False  # TF-IDF by default; sentence-transformers optional

    # --- Unresolved topics ---
    topic_window: int = 10  # sliding-window messages for topic linking
    topic_sim_threshold: float = 0.18  # TF-IDF cosine to group msgs into a topic
    topic_min_shared_tokens: int = 1  # content tokens shared to join a topic
    topic_min_tokens_len: int = 3  # shared tokens must be at least this long
    topic_min_msgs: int = 2  # ignore single-message flukes
    topic_last_msgs: int = 3  # trailing messages treated as "closing" region

    # --- Evidence engine ---
    severity_time_cap_minutes: float = 360.0  # time-unanswered contribution saturates here
    severity_max_reasks: int = 3
    min_severity: float = 0.40  # drop gaps below this after scoring (tuned)
    dedupe_overlap_frac: float = 0.5  # same-type gaps sharing >=50% ids are merged

    # --- Optional embeddings ---
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # --- Optional LLM verification (Gemini) ---
    gemini_model: str = "gemini-2.0-flash"
    gemini_temperature: float = 0.1
    gemini_max_gaps: int = 25

    # --- Cue phrases ---
    answer_cues: tuple[str, ...] = (
        "yes", "yeah", "yep", "no", "nope", "sure", "sounds good", "agreed",
        "done", "fixed", "on it", "will do", "working on it", "here is",
        "here's", "it is", "it's", "because", "i think", "we can", "you can",
        "already", "sent", "check this", "see this", "thanks",
    )
    # Strong answers count on their own (with a little content);
    # weak acknowledgements ("will do", "on it") need content overlap with the question.
    strong_answer_cues: tuple[str, ...] = (
        "yes", "yeah", "yep", "no", "nope", "sure", "agreed", "done",
        "fixed", "sent", "here is", "here's", "already", "sounds good",
        "i can", "i will", "let me", "will send", "will do", "on it",
        "they said", "he said", "she said", "they told me", "working on it",
        "i guess", "supposed to", "i think it was", "i moved", "i checked",
    )
    # Tokens that are too generic to count as topical overlap in answers
    overlap_stop: tuple[str, ...] = (
        "tomorrow", "today", "yesterday", "monday", "tuesday", "wednesday",
        "thursday", "friday", "saturday", "sunday", "tonight", "morning",
        "evening", "night", "week", "month", "soon", "later", "still",
        "just", "really", "actually", "honestly", "sorry", "thanks",
        "please", "okay", "yeah", "yes", "sure", "fine", "okay",
    )
    clarification_cues: tuple[str, ...] = (
        "what do you mean", "can you clarify", "could you clarify",
        "didn't get", "dont get", "don't get", "still not clear",
        "not clear", "i don't understand", "dont understand", "don't understand",
        "can you explain", "what does that mean", "could you elaborate",
        "i'm confused", "im confused", "not following", "huh?",
    )
    closure_cues: tuple[str, ...] = (
        "done", "decided", "let's go with", "lets go with", "agreed",
        "sounds good", "final", "confirmed", "closed", "resolved",
        "we'll go with", "we will go with", "go with", "settled",
        "wrap up", "wrapping up", "approved", "ship it", "case closed",
    )
    social_close_cues: tuple[str, ...] = (
        "thanks", "thank you", "thx", "gn", "good night", "goodnight",
        "noted", "cool", "bye", "cya", "night all", "night",
    )
    decision_lexicon: tuple[str, ...] = (
        "decide", "decided", "should", "need to", "which", "what about",
        "option", "choose", "either", " vs ", "versus", "budget",
        "plan", "schedule", "confirm", "figure out", "still open",
        "never decided", "not decided", "undecided", "deadline",
        "hours", "agenda", "venue", "scope", "owner", "owns",
        "who is", "who takes", "who will", "action item", "follow up",
        "follow-up", "blocker", "blocked",
    )
    deadline_cues: tuple[str, ...] = (
        "deadline", "asap", "urgent", "tomorrow", "today", "friday",
        "by mon", "by monday", "due", "eod", "end of day", "blocking",
        "critical", "as soon as possible", "need this", "required",
    )
    disagreement_cues: tuple[str, ...] = (
        "i disagree", "disagree", "i don't think", "dont think",
        "not convinced", "pushback", "that won't work", "that wont work",
        "i'm against", "im against", "object", "i have concerns",
        "not a fan", "rather not",
    )


DEFAULT_CONFIG = DetectorConfig()
