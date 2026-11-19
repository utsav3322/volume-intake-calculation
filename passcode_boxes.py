"""Passcode boxes: one box per character, like a one-time-code field (a Streamlit component).

Each box takes one letter or digit (A–Z, a–z, 0–9; letters keep their case). Typing moves to
the next box, Backspace goes back, and pasting the whole code fills every box. Nothing is sent
while typing: once the last box is filled (or Enter is pressed with every box filled) the code
is sent once, the boxes clear, and ``app.py`` checks it with ``cowmonitor.access.is_passcode``.
"""

from __future__ import annotations

import streamlit as st

_CSS = """
.passcode {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  font-family: var(--st-font, inherit);
  color: var(--st-text-color, inherit);
}
.passcode-label {
  margin: 0;
  font-size: 0.95rem;
}
.passcode-row {
  display: flex;
  gap: 0.75rem;
}
.passcode-box {
  box-sizing: border-box;
  width: 3.25rem;
  height: 3.75rem;
  padding: 0;
  text-align: center;
  font-family: inherit;
  font-size: 1.6rem;
  color: var(--st-text-color, inherit);
  background: var(--st-secondary-background-color, transparent);
  border: 1px solid var(--st-border-color, rgba(128, 128, 128, 0.45));
  border-radius: var(--st-base-radius, 0.5rem);
  caret-color: var(--st-primary-color, currentColor);
  outline: none;
}
.passcode-box:focus {
  border-color: var(--st-primary-color, #ff4b4b);
  box-shadow: 0 0 0 1px var(--st-primary-color, #ff4b4b);
}
"""

_JS = """
export default function (component) {
  const { data, setTriggerValue, parentElement } = component;
  const length = Number(data && data.length) || 4;
  const allowed = /[A-Za-z0-9]/g;
  const built = parentElement.querySelector(".passcode");
  if (built) {
    if (built.querySelectorAll(".passcode-box").length === length) {
      return;  // Already built: keep what is typed and where the cursor is.
    }
    built.remove();
  }

  const wrapper = document.createElement("div");
  wrapper.className = "passcode";
  const label = document.createElement("p");
  label.className = "passcode-label";
  label.id = "passcode-label";
  label.textContent = `Enter the ${length}-character passcode (letters or numbers)`;
  const row = document.createElement("div");
  row.className = "passcode-row";
  row.setAttribute("role", "group");
  row.setAttribute("aria-labelledby", "passcode-label");
  wrapper.append(label, row);
  // Append, don't replace: the component's stylesheet lives in this same parent.
  parentElement.append(wrapper);

  const boxes = Array.from({ length }, (_, i) => {
    const box = document.createElement("input");
    box.className = "passcode-box";
    box.type = "text";
    box.inputMode = "text";
    box.autocomplete = i === 0 ? "one-time-code" : "off";
    box.autocapitalize = "off";  // phones would otherwise capitalise the first letter
    box.spellcheck = false;
    box.setAttribute("autocorrect", "off");
    box.setAttribute("aria-label", `Character ${i + 1} of ${length}`);
    row.append(box);
    return box;
  });

  const focusBox = (i) => {
    const box = boxes[Math.max(0, Math.min(i, length - 1))];
    box.focus();
    box.select();
  };

  const submitIfFull = () => {
    const code = boxes.map((box) => box.value).join("");
    if (code.length !== length) return;
    setTriggerValue("submitted", code);
    boxes.forEach((box) => { box.value = ""; });
    focusBox(0);
  };

  // Put letters and digits into the boxes from `start` on, then move on and send if every box
  // is filled. Anything else (spaces, symbols) is dropped.
  const fill = (start, text) => {
    const found = (text.match(allowed) || []).slice(0, length - start);
    found.forEach((character, j) => { boxes[start + j].value = character; });
    focusBox(start + found.length);
    submitIfFull();
  };

  boxes.forEach((box, i) => {
    box.addEventListener("focus", () => box.select());
    box.addEventListener("input", (event) => {
      const typed = event.inputType === "insertText" ? event.data || "" : box.value;
      box.value = "";
      if ((typed.match(allowed) || []).length > 0) {
        fill(i, typed);
      }
    });
    box.addEventListener("paste", (event) => {
      event.preventDefault();
      fill(i, (event.clipboardData && event.clipboardData.getData("text")) || "");
    });
    box.addEventListener("keydown", (event) => {
      if (event.key === "Backspace" && !box.value && i > 0) {
        event.preventDefault();
        boxes[i - 1].value = "";
        focusBox(i - 1);
      } else if (event.key === "ArrowLeft") {
        event.preventDefault();
        focusBox(i - 1);
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        focusBox(i + 1);
      } else if (event.key === "Enter") {
        submitIfFull();
      }
    });
  });

  focusBox(0);
}
"""

_component = st.components.v2.component("passcode_boxes", css=_CSS, js=_JS)


def passcode_boxes(*, key: str, length: int) -> str | None:
    """The code typed into ``length`` boxes, on the run right after the last box is filled."""
    result = _component(key=key, data={"length": length}, on_submitted_change=lambda: None)
    return submitted_code(result.get("submitted"))


def submitted_code(value: object) -> str | None:
    """What the browser sent, if it is text; anything else is ignored."""
    return value if isinstance(value, str) else None
