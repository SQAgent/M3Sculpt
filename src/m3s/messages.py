"""Plain-dict conversation messages. No LangChain message classes."""


def human(text):
    """Build a user message."""
    return {"role": "user", "content": text}


def ai(text=None, blocks=None):
    """Build an assistant message with either plain text or content blocks."""
    if blocks is not None:
        return {"role": "assistant", "content": list(blocks)}
    return {"role": "assistant", "content": text or ""}


def is_human(message):
    return message.get("role") == "user"


def text_of(message):
    """Flatten a message's content down to concatenated text."""
    content = message.get("content") if message else ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(block.get("text", "") for block in content
                        if isinstance(block, dict) and block.get("type") == "text")
    return ""


def last_human_text(messages):
    """Return the text of the most recent user message."""
    return next((text_of(m) for m in reversed(messages or []) if is_human(m)), "")


def first_human_text(messages):
    """Return the text of the first user message in the conversation."""
    return next((text_of(m) for m in messages or [] if is_human(m)), "")
