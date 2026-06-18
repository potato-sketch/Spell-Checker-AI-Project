const inputText = document.getElementById("input-text");
const wordCount = document.getElementById("word-count");
const charCount = document.getElementById("char-count");
const suggCount = document.getElementById("sugg-count");
const copyBtn = document.getElementById("copy-btn");
const deleteBtn = document.getElementById("delete-btn");
const checkBtn = document.getElementById("check-btn");
const toastMessage = document.getElementById("toast-message");
const suggestionsList = document.getElementById("suggestions-list");
const emptyState = document.getElementById("empty-state");

let toastTimer = null;
let liveCheckTimer = null;
let activeCheckId = 0;
let lastCheckedText = "";

function showToast(message) {
  toastMessage.textContent = message;
  toastMessage.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toastMessage.classList.remove("show");
  }, 1800);
}

function countWords(text) {
  const trimmed = text.trim();
  return trimmed === "" ? 0 : trimmed.split(/\s+/).filter(Boolean).length;
}

function renderSuggestions(errors) {
  suggestionsList.innerHTML = "";

  if (!errors.length) {
    suggestionsList.style.display = "none";
    emptyState.style.display = "block";
    suggCount.textContent = "0";
    return;
  }

  suggestionsList.style.display = "block";
  emptyState.style.display = "none";
  suggCount.textContent = String(errors.length);

  const label = document.createElement("p");
  label.className = "suggestions-label";
  label.textContent = "Other Suggestions:";
  suggestionsList.appendChild(label);

  errors.forEach((error) => {
    const item = document.createElement("div");
    item.className = "suggestion-item";
    item.innerHTML = `
      <strong>${error.token}</strong> - ${error.reason}<br />
      ${error.suggestions.length ? error.suggestions.join(", ") : "No suggestion"}
    `;
    suggestionsList.appendChild(item);
  });
}

function updateCounters() {
  const text = inputText.value;
  const words = countWords(text);

  if (words > 500) {
    inputText.value = inputText.value.trim().split(/\s+/).slice(0, 500).join(" ");
  }

  const cappedWords = Math.min(words, 500);
  wordCount.textContent = cappedWords;
  charCount.textContent = inputText.value.length;
}

async function checkText(options = {}) {
  const { source = "manual", silent = false } = options;
  const text = inputText.value.trim();
  if (!text) {
    if (!silent) {
      showToast("Please type some text first.");
    }
    renderSuggestions([]);
    lastCheckedText = "";
    return;
  }

  if (source === "live" && text === lastCheckedText) {
    return;
  }

  const checkId = ++activeCheckId;
  if (source === "manual") {
    checkBtn.disabled = true;
    checkBtn.textContent = "Checking...";
  }

  try {
    const response = await fetch("/api/check", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      throw new Error("Request failed");
    }

    const data = await response.json();
    if (checkId !== activeCheckId) {
      return;
    }

    renderSuggestions(data.errors || []);
    suggCount.textContent = String(data.error_count || 0);
    lastCheckedText = text;

    // popup/toast when no errors found (real-time supported via silent flag)
    if (!silent && source === "manual") {
      showToast(data.error_count ? "Errors detected." : "No errors detected!");
    }
    if (!data.error_count && source !== "manual") {
      // keep UI clean for live mode; still update an unobtrusive toast
      // only if suggestions area is currently visible
      // (avoids spamming users)
    }
  } catch (error) {
    if (checkId !== activeCheckId) {
      return;
    }
    if (!silent && source === "manual") {
      showToast("Backend check failed.");
    }
  } finally {
    if (checkId === activeCheckId && source === "manual") {
      checkBtn.disabled = false;
      checkBtn.textContent = "Check Text";
    }
  }
}

function scheduleLiveCheck() {
  clearTimeout(liveCheckTimer);
  liveCheckTimer = setTimeout(() => {
    checkText({ source: "live", silent: true });
  }, 350);
}

inputText.addEventListener("input", function () {
  updateCounters();
  // Real-time feature disabled: wait for manual Check
});

deleteBtn.addEventListener("click", function () {
  inputText.value = "";
  wordCount.textContent = "0";
  charCount.textContent = "0";
  suggCount.textContent = "0";
  suggestionsList.innerHTML = "";
  suggestionsList.style.display = "none";
  emptyState.style.display = "block";
  inputText.focus();
  showToast("Text cleared.");
});

copyBtn.addEventListener("click", async function () {
  const text = inputText.value.trim();
  if (!text) {
    showToast("Nothing to copy.");
    return;
  }

  try {
    await navigator.clipboard.writeText(inputText.value);
    showToast("Text copied successfully.");
  } catch (error) {
    showToast("Copy failed.");
  }
});

// Run in real-time via the input event; keep button hidden but functional if needed.
checkBtn.style.display = "inline-flex";
checkBtn.addEventListener("click", () => checkText({ source: "manual", silent: false }));


suggestionsList.style.display = "none";
emptyState.style.display = "block";
updateCounters();
