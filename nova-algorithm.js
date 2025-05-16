/* ======================================================================
   nova-algorithm.js  ·  v4: Enhanced NLP, Robust Context, UI Integration
   ====================================================================== */

/**
 * NovaAI: A client-side AI for processing natural language queries with context-aware responses
 * and seamless integration with the provided HTML interface and JSON knowledge base.
 * Features: Advanced intent detection, sentiment analysis, entity extraction, history tracking,
 * and dynamic UI updates for search, history, and bookmarks.
 */
(function (global) {

/* ---------- Text Processing Utilities ---------- */
const STOP_WORDS = new Set([
  "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
  "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "with",
  "what", "which", "when", "where", "how", "why"
]);

const ALIAS = {
  ai: "artificial intelligence",
  llm: "large language model",
  ml: "machine learning",
  iot: "internet of things",
  nlp: "natural language processing",
  vr: "virtual reality",
  ar: "augmented reality",
  crypto: "blockchain",
  wasm: "webassembly",
  cbt: "cognitive behavioral therapy"
};

const SYNONYMS = {
  blockchain: ["crypto", "distributed ledger"],
  ai: ["machine learning", "neural network"],
  quantum: ["qubit", "superposition"],
  climate: ["weather", "environment"],
  coding: ["programming", "software"],
  ethics: ["morality", "philosophy"]
};

const SIMPLE_LEMMA = {
  running: "run", runs: "run", ran: "run",
  coded: "code", coding: "code", codes: "code",
  technologies: "technology", models: "model",
  facts: "fact", learned: "learn", learning: "learn"
};

/**
 * Expands a word using aliases.
 * @param {string} word - Input word.
 * @returns {string[]} - Array of expanded words.
 */
const expand = word => ALIAS[word] ? ALIAS[word].split(" ") : [word];

/**
 * Lemmatizes a word to its base form.
 * @param {string} word - Input word.
 * @returns {string} - Lemmatized word.
 */
const lemmatize = word => SIMPLE_LEMMA[word] || word.replace(/s$/, "");

/**
 * Tokenizes text into words, applying lemmatization and synonym expansion.
 * @param {string} text - Input text.
 * @returns {string[]} - Array of tokens.
 */
const tokens = text => {
  const out = [];
  text.toLowerCase().split(/[^a-z0-9]+/).forEach(word => {
    if (!word || STOP_WORDS.has(word)) return;
    const lemma = lemmatize(word);
    out.push(...expand(lemma));
    if (SYNONYMS[lemma]) out.push(...SYNONYMS[lemma].map(lemmatize));
  });
  return out;
};

/**
 * Creates a bag-of-words from an array of tokens.
 * @param {string[]} arr - Array of tokens.
 * @returns {Object} - Frequency map.
 */
const bag = arr => arr.reduce((obj, word) => {
  obj[word] = (obj[word] || 0) + 1;
  return obj;
}, {});

/**
 * Generates n-grams from tokens.
 * @param {string[]} arr - Array of tokens.
 * @param {number} n - N-gram size.
 * @returns {string[]} - Array of n-grams.
 */
const ngrams = (arr, n) => {
  const out = [];
  for (let i = 0; i <= arr.length - n; i++) {
    out.push(arr.slice(i, i + n).join("_"));
  }
  return out;
};

/**
 * Computes cosine similarity between two vectors.
 * @param {Object} a - First vector.
 * @param {Object} b - Second vector.
 * @returns {number} - Cosine similarity score.
 */
const cos = (a, b) => {
  let dot = 0, magA = 0, magB = 0;
  for (const k in a) {
    magA += a[k] ** 2;
    if (b[k]) dot += a[k] * b[k];
  }
  for (const k in b) magB += b[k] ** 2;
  return dot ? dot / Math.sqrt(magA * magB) : 0;
};

/**
 * Returns unique elements from an array.
 * @param {string[]} arr - Input array.
 * @returns {string[]} - Unique elements.
 */
const uniq = arr => [...new Set(arr)];

/**
 * Shuffles an array randomly.
 * @param {string[]} arr - Input array.
 * @returns {string[]} - Shuffled array.
 */
const shuffle = arr => arr.slice().sort(() => 0.5 - Math.random());

/* ---------- NovaAI Class ---------- */
class NovaAI {
  /**
   * Initializes NovaAI with knowledge base and UI bindings.
   * @param {string} kbSrc - Path to knowledge base JSON.
   * @param {Object} uiConfig - UI element selectors.
   */
  constructor(kbSrc = "kb.json", uiConfig = {}) {
    // Default UI selectors
    this.ui = {
      searchInput: uiConfig.searchInput || "#search-input",
      searchButton: uiConfig.searchButton || "#search-button",
      clearButton: uiConfig.clearButton || "#clear-button",
      resultArea: uiConfig.resultArea || "#results",
      historyArea: uiConfig.historyArea || "#history-list",
      savedArea: uiConfig.savedArea || "#saved-list",
      chipArea: uiConfig.chipArea || "#chips",
      ...uiConfig
    };

    // Load knowledge base synchronously
    try {
      const xhr = new XMLHttpRequest();
      xhr.open("GET", kbSrc, false);
      xhr.send(null);
      this.kb = JSON.parse(xhr.responseText).facts;
    } catch (e) {
      console.error("Failed to load knowledge base:", e);
      this.kb = [];
    }

    // Preprocess knowledge base
    this.df = {};
    this.N = this.kb.length;
    this.kb.forEach(fact => {
      fact.toks = tokens(fact.text);
      fact.bow = bag([...fact.toks, ...tokens(fact.keywords.join(" "))]);
      fact.ngrams = [...ngrams(fact.toks, 2), ...ngrams(fact.toks, 3)];
      fact.ngramBow = bag(fact.ngrams);
      for (const k in fact.bow) this.df[k] = (this.df[k] || 0) + 1;
      for (const k in fact.ngramBow) this.df[k] = (this.df[k] || 0) + 1;
    });

    // Compute TF-IDF vectors
    this.idf = {};
    for (const k in this.df) this.idf[k] = Math.log(this.N / (this.df[k] || 1));
    this.kb.forEach(fact => {
      const vec = {};
      for (const k in fact.bow) vec[k] = fact.bow[k] * this.idf[k];
      for (const k in fact.ngramBow) vec[k] = fact.ngramBow[k] * (this.idf[k] || 1);
      const norm = Math.sqrt(Object.values(vec).reduce((s, x) => s + x * x, 0)) || 1;
      for (const k in vec) vec[k] /= norm;
      fact.vec = vec;
    });

    // Cache topics and subtopics
    this.topics = uniq(this.kb.map(f => f.topic));
    this.subtopics = uniq(this.kb.flatMap(f => f.subtopics));

    // Context and history
    this.context = {
      currentTopic: null,
      lastTopic: null,
      history: [],
      saved: JSON.parse(localStorage.getItem("nova-saved") || "[]"),
      maxHistory: 50
    };

    // Response templates by intent
    this.templates = {
      question: {
        main: "{sentiment} You asked about {entity} in {topic}. {content}",
        followUp: "Want to explore more on {entity} or check out {relatedTopic}?"
      },
      explanation: {
        main: "{sentiment} Here's how {entity} works in {topic}: {content}. It's driven by {mechanism}.",
        followUp: "Curious about {entity}'s impact or another {topic} topic?"
      },
      description: {
        main: "{sentiment} {entity} in {topic} is fascinating: {content}. It shapes {impact}.",
        followUp: "Interested in deeper insights on {entity} or related {topic} areas?"
      },
      clarification: {
        main: "{sentiment} I'm not sure about '{query}'. Could you clarify? Try asking about {topic} or {entity}.",
        followUp: "E.g., 'What is {entity}?' or 'How does {topic} work?'"
      },
      comparison: {
        main: "{sentiment} Comparing {entity} and {entity2} in {topic}: {content}. They differ in {difference}.",
        followUp: "Want to dive deeper into {entity} or compare another {topic} aspect?"
      }
    };

    // Sentiment phrases
    this.sentiments = {
      positive: ["Great question!", "Love the enthusiasm!", "Awesome topic!"],
      curious: ["That's a deep one!", "Curious angle, let's dive in!", "Intriguing query!"],
      neutral: ["Got it, let's explore.", "Here's the scoop.", "Alright, let's tackle it."],
      confused: ["Let's clear that up!", "I hear you, let's sort it out.", "No worries, I'll clarify."]
    };

    // Initialize UI bindings
    this.initUI();
  }

  /**
   * Initializes UI event listeners and loads history/saved items.
   */
  initUI() {
    const searchInput = document.querySelector(this.ui.searchInput);
    const searchButton = document.querySelector(this.ui.searchButton);
    const clearButton = document.querySelector(this.ui.clearButton);
    const historyArea = document.querySelector(this.ui.historyArea);
    const savedArea = document.querySelector(this.ui.savedArea);

    if (searchInput && searchButton) {
      searchButton.addEventListener("click", () => this.handleSearch());
      searchInput.addEventListener("keypress", e => {
        if (e.key === "Enter") this.handleSearch();
      });
    }

    if (clearButton) {
      clearButton.addEventListener("click", () => {
        searchInput.value = "";
        this.renderResults(this.blank());
      });
    }

    // Load history and saved items
    this.renderHistory();
    this.renderSaved();

    // Tab switching
    document.querySelectorAll(".tab").forEach(tab => {
      tab.addEventListener("click", () => this.switchTab(tab.dataset.tab));
    });
  }

  /**
   * Handles search button click or Enter key press.
   */
  handleSearch() {
    const searchInput = document.querySelector(this.ui.searchInput);
    const query = searchInput.value.trim();
    if (query) {
      const response = this.processQuery(query);
      this.renderResults(response);
      this.context.history.push({ query, ...response, ts: Date.now() });
      if (this.context.history.length > this.context.maxHistory) {
        this.context.history.shift();
      }
      this.renderHistory();
      localStorage.setItem("nova-history", JSON.stringify(this.context.history));
    }
  }

  /**
   * Processes a user query and returns a structured response.
   * @param {string} query - User's input.
   * @returns {object} - { topic, main, details, stats, quote, related }
   */
  processQuery(query) {
    const q = query.trim();
    if (!q) return this.blank();

    // Parse query
    const { intent, sentiment, entities, qVec } = this.parseQuery(q);

    // Rank facts
    const ranked = this.kb.map(fact => {
      const sim = cos(qVec, fact.vec);
      const topicBoost = this.context.currentTopic === fact.topic ? 1.3 : 1;
      const historyBoost = this.context.history.slice(-5).some(h => h.topic === fact.topic) ? 1.2 : 1;
      const subtopicBoost = entities.some(e => fact.subtopics.includes(e.keyword)) ? 1.1 : 1;
      return { ...fact, score: sim * fact.weight * topicBoost * historyBoost * subtopicBoost };
    }).filter(r => r.score > 0.15)
      .sort((a, b) => b.score - a.score);

    if (!ranked.length) return this.sorry(q);

    // Compose response
    const topic = ranked[0].topic;
    const topFacts = ranked.slice(0, 3);
    const main = this.composeResponse(topFacts, intent, sentiment, entities, topic, q);
    const details = ranked.slice(3, 8).map(f => ({
      text: f.text,
      id: `${f.topic}-${f.keywords.join("-")}`
    }));
    const related = this.generateChips(topic, entities, q, ranked);

    // Update context
    this.context.lastTopic = this.context.currentTopic;
    this.context.currentTopic = topic;
    return { topic, main, details, stats: this.computeStats(ranked), quote: null, related };
  }

  /**
   * Parses query for intent, sentiment, entities, and vector.
   * @param {string} query - User's input.
   * @returns {object} - { intent, sentiment, entities, qVec }
   */
  parseQuery(query) {
    const qTok = tokens(query);
    const qBow = bag([...qTok, ...ngrams(qTok, 2), ...ngrams(qTok, 3)]);
    const qVec = {};
    for (const k in qBow) if (this.idf[k]) qVec[k] = qBow[k] * this.idf[k];
    const norm = Math.sqrt(Object.values(qVec).reduce((s, x) => s + x * x, 0)) || 1;
    for (const k in qVec) qVec[k] /= norm;

    // Intent detection
    const lowerQ = query.toLowerCase();
    let intent = "description";
    const isAmbiguous = lowerQ.length < 5 || ["it", "this", "that"].includes(lowerQ);
    const intentPatterns = {
      question: /^(what|how|why|when|where|who)\b|.*\?$/,
      explanation: /\b(how|works|process|explain|mechanism)\b/,
      description: /\b(what|about|describe|is|tell)\b/,
      comparison: /\b(compare|versus|vs|difference|similar)\b/,
      clarification: /\b(it|this|that)\b|^.{0,4}$/
    };
    for (const [key, pattern] of Object.entries(intentPatterns)) {
      if (pattern.test(lowerQ)) {
        intent = key;
        break;
      }
    }
    if (isAmbiguous && intent !== "question") intent = "clarification";

    // Sentiment detection
    let sentiment = "neutral";
    const sentimentKeywords = {
      positive: ["great", "awesome", "excited", "love", "amazing"],
      curious: ["wonder", "curious", "why", "how", "intriguing"],
      confused: ["confused", "unclear", "not sure", "help", "clarify"]
    };
    for (const [key, keywords] of Object.entries(sentimentKeywords)) {
      if (keywords.some(k => lowerQ.includes(k))) {
        sentiment = key;
        break;
      }
    }

    // Entity extraction
    const entities = [];
    this.kb.forEach(fact => {
      fact.keywords.forEach(k => {
        if (lowerQ.includes(k.toLowerCase()) && !entities.some(e => e.keyword === k)) {
          entities.push({ topic: fact.topic, keyword: k, weight: k.length / lowerQ.length });
        }
      });
    });
    if (this.context.history.length) {
      const recentEntities = this.context.history.slice(-5).flatMap(h => h.entities || []);
      entities.push(...recentEntities.filter(e => !entities.some(ex => ex.keyword === e.keyword)));
    }

    return { intent, sentiment, entities, qVec };
  }

  /**
   * Composes the main response using top facts and context.
   * @param {Array} facts - Top-ranked facts.
   * @param {string} intent - Detected intent.
   * @param {string} sentiment - Detected sentiment.
   * @param {Array} entities - Extracted entities.
   * @param {string} topic - Selected topic.
   * @param {string} query - Original query.
   * @returns {string} - Formatted response.
   */
  composeResponse(facts, intent, sentiment, entities, topic, query) {
    const template = this.templates[intent] || this.templates.description;
    const entity = entities[0]?.keyword || topic;
    const entity2 = entities[1]?.keyword || this.topics.find(t => t !== topic) || topic;
    const isFollowUp = this.context.history.some(h => h.topic === topic);
    const topicShift = this.context.lastTopic && this.context.lastTopic !== topic;

    let response = "";
    if (topicShift) response += `Switching focus to ${topic}. `;
    else if (isFollowUp) response += `Continuing on ${topic}. `;
    
    response += this.sentiments[sentiment][Math.floor(Math.random() * this.sentiments[sentiment].length)] + " ";

    const content = facts.length > 1 
      ? `${facts[0].text} Also, ${facts[1].text}${facts[2] ? `. Plus, ${facts[2].text}` : ""}`
      : facts[0].text;

    const difference = facts.length > 1 ? `${facts[0].keywords[0]} focuses on ${facts[0].subtopics[0]}, while ${facts[1].keywords[0]} emphasizes ${facts[1].subtopics[0]}` : "their approach";

    response += template.main
      .replace("{sentiment}", "")
      .replace("{entity}", entity)
      .replace("{entity2}", entity2)
      .replace("{topic}", topic)
      .replace("{content}", content)
      .replace("{mechanism}", facts[0].text.split(".")[0] || "key processes")
      .replace("{impact}", facts[0].subtopics[0] || "its field")
      .replace("{difference}", difference)
      .replace("{query}", query);

    response += " " + template.followUp
      .replace("{entity}", entity)
      .replace("{topic}", topic)
      .replace("{relatedTopic}", this.topics.find(t => t !== topic) || "another area");

    return response;
  }

  /**
   * Generates follow-up query chips.
   * @param {string} topic - Current topic.
   * @param {Array} entities - Extracted entities.
   * @param {string} query - Original query.
   * @param {Array} ranked - Ranked facts.
   * @returns {Array} - List of chip strings.
   */
  generateChips(topic, entities, query, ranked) {
    const semantic = ranked.slice(3, 10).map(f => {
      const t = f.text.split(".")[0];
      return t.length > 50 ? t.slice(0, 47) + "…" : t;
    });
    const refine = uniq(tokens(query)).slice(0, 4).map(w => `More on ${w}?`);
    const stock = [
      `What’s new in ${topic}?`,
      `How does ${topic} work?`,
      `Key trends in ${topic}`,
      `Challenges in ${topic}`,
      `Future of ${topic}`,
      `Compare ${topic} to ${this.context.lastTopic || "another area"}`
    ];
    const cross = this.context.lastTopic && this.context.lastTopic !== topic
      ? [`How does ${topic} relate to ${this.context.lastTopic}?`]
      : [];
    const entityChips = entities.slice(0, 3).map(e => `Deep dive on ${e.keyword}?`);

    return shuffle(uniq([...semantic, ...refine, ...stock, ...cross, ...entityChips])).slice(0, 6);
  }

  /**
   * Computes statistics for the response.
   * @param {Array} ranked - Ranked facts.
   * @returns {Object} - Statistics object.
   */
  computeStats(ranked) {
    const topTopics = uniq(ranked.slice(0, 10).map(f => f.topic));
    return {
      confidence: ranked[0]?.score.toFixed(2) || 0,
      topicsCovered: topTopics.length,
      topTopic: topTopics[0] || "N/A"
    };
  }

  /**
   * Renders the response to the UI.
   * @param {Object} response - Response object.
   */
  renderResults(response) {
    const resultArea = document.querySelector(this.ui.resultArea);
    const chipArea = document.querySelector(this.ui.chipArea);
    if (!resultArea || !chipArea) return;

    resultArea.innerHTML = `
      <div class="result-card p-4 bg-white dark:bg-gray-800 rounded-lg shadow-md">
        <h2 class="text-xl font-semibold text-gray-900 dark:text-white">${response.topic || "General"}</h2>
        <p class="mt-2 text-gray-700 dark:text-gray-300">${response.main}</p>
        ${response.details.length ? `
          <h3 class="mt-4 text-lg font-medium text-gray-900 dark:text-white">More Details</h3>
          <ul class="mt-2 space-y-2">
            ${response.details.map(d => `
              <li class="flex items-start">
                <span class="text-gray-700 dark:text-gray-300">${d.text}</span>
                <button class="ml-2 text-blue-500 hover:text-blue-700 save-btn" data-id="${d.id}">Save</button>
              </li>
            `).join("")}
          </ul>
        ` : ""}
        ${response.stats ? `
          <div class="mt-4 text-sm text-gray-500 dark:text-gray-400">
            Confidence: ${response.stats.confidence} | Topics: ${response.stats.topicsCovered}
          </div>
        ` : ""}
      </div>
    `;

    chipArea.innerHTML = response.related.map(chip => `
      <button class="chip bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 px-3 py-1 rounded-full mr-2 mb-2 hover:bg-gray-300 dark:hover:bg-gray-600">${chip}</button>
    `).join("");

    // Add event listeners for chips
    document.querySelectorAll(".chip").forEach(chip => {
      chip.addEventListener("click", () => {
        document.querySelector(this.ui.searchInput).value = chip.textContent.replace("?", "");
        this.handleSearch();
      });
    });

    // Add event listeners for save buttons
    document.querySelectorAll(".save-btn").forEach(btn => {
      btn.addEventListener("click", () => this.saveResult(btn.dataset.id, response));
    });
  }

  /**
   * Saves a result to local storage and updates saved list.
   * @param {string} id - Result ID.
   * @param {Object} response - Response object.
   */
  saveResult(id, response) {
    const result = response.details.find(d => d.id === id);
    if (result && !this.context.saved.some(s => s.id === id)) {
      this.context.saved.push({ id, text: result.text, topic: response.topic, ts: Date.now() });
      localStorage.setItem("nova-saved", JSON.stringify(this.context.saved));
      this.renderSaved();
    }
  }

  /**
   * Renders the history list.
   */
  renderHistory() {
    const historyArea = document.querySelector(this.ui.historyArea);
    if (!historyArea) return;

    historyArea.innerHTML = this.context.history.reverse().map(h => `
      <div class="history-item p-2 bg-gray-100 dark:bg-gray-700 rounded-md mb-2 cursor-pointer hover:bg-gray-200 dark:hover:bg-gray-600">
        <span class="text-gray-800 dark:text-gray-200">${h.query}</span>
        <span class="text-sm text-gray-500 dark:text-gray-400">(${h.topic})</span>
      </div>
    `).join("");

    document.querySelectorAll(".history-item").forEach(item => {
      item.addEventListener("click", () => {
        document.querySelector(this.ui.searchInput).value = item.querySelector("span").textContent;
        this.handleSearch();
      });
    });
  }

  /**
   * Renders the saved items list.
   */
  renderSaved() {
    const savedArea = document.querySelector(this.ui.savedArea);
    if (!savedArea) return;

    savedArea.innerHTML = this.context.saved.map(s => `
      <div class="saved-item p-2 bg-gray-100 dark:bg-gray-700 rounded-md mb-2">
        <span class="text-gray-800 dark:text-gray-200">${s.text}</span>
        <span class="text-sm text-gray-500 dark:text-gray-400">(${s.topic})</span>
        <button class="ml-2 text-red-500 hover:text-red-700 remove-btn" data-id="${s.id}">Remove</button>
      </div>
    `).join("");

    document.querySelectorAll(".remove-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        this.context.saved = this.context.saved.filter(s => s.id !== btn.dataset.id);
        localStorage.setItem("nova-saved", JSON.stringify(this.context.saved));
        this.renderSaved();
      });
    });
  }

  /**
   * Switches between tabs (All, Recent, Bookmarked).
   * @param {string} tab - Tab name.
   */
  switchTab(tab) {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelector(`[data-tab="${tab}"]`).classList.add("active");
    document.querySelectorAll(".tab-content").forEach(content => content.classList.add("hidden"));
    document.querySelector(`#${tab.toLowerCase()}-content`).classList.remove("hidden");
  }

  /**
   * Returns a blank response for empty queries.
   * @returns {object} - Blank response object.
   */
  blank() {
    return {
      main: "Ask me anything!",
      details: [],
      stats: null,
      quote: null,
      related: []
    };
  }

  /**
   * Returns a fallback response for unmatched queries.
   * @param {string} query - Original query.
   * @returns {object} - Fallback response object.
   */
  sorry(query) {
    return {
      main: `Sorry, I don’t have info on "${query}". Try asking about ${this.topics.slice(0, 3).join(", ")}!`,
      details: [],
      stats: null,
      quote: null,
      related: this.topics.slice(0, 3).map(t => `What is ${t}?`)
    };
  }
}

/* ---------- Export and Initialize ---------- */
global.NovaAI = NovaAI;

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
  const nova = new NovaAI("kb.json", {
    searchInput: "#search-input",
    searchButton: "#search-button",
    clearButton: "#clear-button",
    resultArea: "#results",
    historyArea: "#history-list",
    savedArea: "#saved-list",
    chipArea: "#chips"
  });
});

})(typeof window !== "undefined" ? window : this);