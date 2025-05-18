// nova-algorithm.js  ·  v9.1: Concept Clustering, Session Memory, Intent Detection
/* ====================================================================== */

(function (global) {

/* ---------- Constants and Helper Maps ---------- */
const STOP_WORDS = new Set([
  "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
  "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "with",
  "what", "which", "when", "where", "how", "why"
]);

const SYNONYMS = {
  ai: "artificial intelligence",
  llm: "large language model",
  vr: "virtual reality",
  btc: "bitcoin"
};

const RELATED_TERMS = {
  ai: ["machine-learning", "neural-networks", "deep-learning"],
  blockchain: ["decentralized", "ledger", "cryptocurrency"],
  quantum: ["qubits", "superposition", "entanglement"],
  health: ["wellness", "nutrition", "fitness"],
  environment: ["sustainability", "climate", "ecology"]
};

const CONCEPTS = {
  "future-tech": ["ai", "quantum-tech", "blockchain", "digital-twins", "augmented-reality"],
  "green-tech": ["environment", "biochar", "carbon-capture", "offshore-wind"],
  "data-security": ["zero-trust", "cryptography", "blockchain"],
  "human-augmentation": ["health", "wearable-tech", "augmented-reality", "soft-robotics"]
};

const STEM_SUFFIXES = ["ing", "ed", "es", "s"];
const EMBED_DIM = 100;
const VIEW_DECAY = 0.95;
const MAX_CACHE_SIZE = 1000;

/* ---------- Text Processing Utilities ---------- */
/**
 * Computes Jaccard similarity of weighted trigrams between two strings.
 * @param {string} a - First string.
 * @param {string} b - Second string.
 * @param {Object} idf - IDF weights.
 * @returns {number} - Weighted Jaccard similarity.
 */
const weightedTrigramSimilarity = (a, b, idf) => {
  const aTrigrams = new Map(generateSubwords(a).map(t => [t, idf[t] || 1]));
  const bTrigrams = new Map(generateSubwords(b).map(t => [t, idf[t] || 1]));
  let intersection = 0, union = 0;
  const allTrigrams = new Set([...aTrigrams.keys(), ...bTrigrams.keys()]);
  for (const t of allTrigrams) {
    const aWeight = aTrigrams.get(t) || 0;
    const bWeight = bTrigrams.get(t) || 0;
    intersection += Math.min(aWeight, bWeight);
    union += Math.max(aWeight, bWeight);
  }
  return union ? intersection / union : 0;
};

/**
 * Stems a word by removing common suffixes.
 * @param {string} word - Input word.
 * @returns {string} - Stemmed word.
 */
const stemWord = word => {
  for (const suffix of STEM_SUFFIXES) {
    if (word.endsWith(suffix)) {
      return word.slice(0, -suffix.length);
    }
  }
  return word;
};

/**
 * Splits camelCase words into parts.
 * @param {string} word - Input word.
 * @returns {string[]} - Array of split parts.
 */
const splitCamelCase = word => {
  const parts = [];
  let start = 0;
  for (let i = 1; i < word.length; i++) {
    if (word[i] === word[i].toUpperCase() && word[i - 1] === word[i - 1].toLowerCase()) {
      parts.push(word.slice(start, i));
      start = i;
    }
  }
  parts.push(word.slice(start));
  return parts;
};

/**
 * Smart tokenization: splits text, handles camelCase, snake_case, synonyms, and stemming.
 * @param {string} text - Input text.
 * @returns {string[]} - Array of tokens.
 */
const tokenCache = new Map();
const smartTokenize = text => {
  if (tokenCache.has(text)) return tokenCache.get(text);
  const potentialWords = text.split(/[^a-zA-Z0-9]+/).filter(Boolean);
  const tokens = [];
  potentialWords.forEach(word => {
    let expanded = SYNONYMS[word.toLowerCase()] || word;
    if (expanded.includes(' ')) {
      tokens.push(...expanded.split(' ').map(w => stemWord(w.toLowerCase())));
    } else if (expanded.includes('_')) {
      tokens.push(...expanded.split('_').map(w => stemWord(w)));
    } else if (/[A-Z]/.test(expanded)) {
      tokens.push(...splitCamelCase(expanded).map(w => stemWord(w)));
    } else {
      tokens.push(stemWord(expanded));
    }
  });
  const result = tokens.map(t => t.toLowerCase()).filter(t => !STOP_WORDS.has(t) && t.length > 1);
  tokenCache.set(text, result);
  if (tokenCache.size > MAX_CACHE_SIZE) {
    const oldestKey = tokenCache.keys().next().value;
    tokenCache.delete(oldestKey);
  }
  return result;
};

/**
 * Generates 3-character sub-words from a word.
 * @param {string} word - Input word.
 * @returns {string[]} - Array of sub-words.
 */
const generateSubwords = word => {
  if (word.length <= 3) return [];
  const subwords = [];
  for (let i = 0; i <= word.length - 3; i++) {
    subwords.push(word.slice(i, i + 3));
  }
  return subwords;
};

/**
 * Generates n-grams from tokens.
 * @param {string[]} arr - Array of tokens.
 * @param {number} n - N-gram size.
 * @returns {string[]} - Array of n-grams.
 */
const ngrams = (arr, n) => {
  const out = [];
  for (let i = 0; i <= arr.length - n; i++) {
    out.push(arr.slice(i, i + n).join(" "));
  }
  return out;
};

/**
 * Extracts phrases (2 or 3 words) from tokens.
 * @param {string[]} tokens - Array of tokens.
 * @returns {string[]} - Array of phrases.
 */
const extractPhrases = tokens => {
  return [...ngrams(tokens, 2), ...ngrams(tokens, 3)];
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
 * Computes a simple hash for a string.
 * @param {string} str - Input string.
 * @returns {number} - Hash value.
 */
const simpleHash = str => {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = (hash * 31 + str.charCodeAt(i)) & 0x7fffffff;
  }
  return hash;
};

/**
 * Generates a pseudo-embedding (100-dim) from tokens.
 * @param {string[]} tokens - Array of tokens.
 * @returns {number[]} - Dense vector.
 */
const pseudoEmbed = tokens => {
  const vec = Array(EMBED_DIM).fill(0);
  const trigrams = tokens.flatMap(t => generateSubwords(t));
  trigrams.forEach(tri => {
    const idx = simpleHash(tri) % EMBED_DIM;
    vec[idx] += 1 / (trigrams.length || 1);
  });
  const norm = Math.sqrt(vec.reduce((s, x) => s + x * x, 0)) || 1;
  return vec.map(x => x / norm);
};

/**
 * Computes cosine similarity between two vectors.
 * @param {number[]} a - First vector.
 * @param {number[]} b - Second vector.
 * @returns {number} - Cosine similarity.
 */
const cos = (a, b) => {
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] ** 2;
    magB += b[i] ** 2;
  }
  return dot ? dot / Math.sqrt(magA * magB) : 0;
};

/**
 * Finds common noun phrases (≥ 3 chars) across sentences.
 * @param {string[]} sentences - Array of sentences.
 * @returns {string|null} - Common phrase or null.
 */
const findCommonPhrase = sentences => {
  const phrases = sentences.map(s => extractPhrases(smartTokenize(s)));
  const common = phrases[0].filter(p => p.length >= 3 && phrases.every(ph => ph.includes(p)));
  return common[0] || null;
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

/* ---------- Intent Detection ---------- */
const INTENT_PATTERNS = {
  definition: /^(explain|what is|define)\s+/i,
  comparison: /compare.*to/i,
  list: /(list|top|best).*of/i,
  general: /.*/ // Fallback
};

const responseTemplates = {
  definition: {
    connectors: ["To clarify", "In essence"],
    structure: fact => `${fact.text}<sup>[1]</sup>`,
    support: (text, idx) => `${text}<sup>[${idx}]</sup>`
  },
  comparison: {
    connectors: ["In contrast", "Meanwhile"],
    structure: (fact1, fact2) => `${fact1.text}<sup>[1]</sup> ${fact2 ? `In contrast, ${fact2.text.toLowerCase()}<sup>[2]</sup>` : ""}`,
    support: (text, idx) => `${text}<sup>[${idx}]</sup>`
  },
  list: {
    connectors: ["Also", "Additionally"],
    structure: facts => facts.map((f, i) => `${i + 1}. ${f.text}<sup>[${i + 1}]</sup>`).join(" "),
    support: (text, idx) => `${text}<sup>[${idx}]</sup>`
  },
  general: {
    connectors: ["Additionally", "Furthermore"],
    structure: facts => {
      const common = findCommonPhrase(facts.map(f => f.text));
      if (common) {
        return facts.map((f, i) => `${f.text.replace(new RegExp(common, "i"), common)}<sup>[${i + 1}]</sup>`).join(" Furthermore, ");
      }
      return facts.map((f, i) => `${f.text}<sup>[${i + 1}]</sup>`).join(" Additionally, ");
    },
    support: (text, idx) => `${text.toLowerCase()}<sup>[${idx}]</sup>`
  }
};

/**
 * Detects query intent.
 * @param {string} query - User query.
 * @returns {string} - Intent type.
 */
const detectIntent = query => {
  for (const [intent, pattern] of Object.entries(INTENT_PATTERNS)) {
    if (pattern.test(query)) return intent;
  }
  return "general";
};

/* ---------- NovaAI Class ---------- */
class NovaAI {
  /**
   * Initializes NovaAI with knowledge base and UI bindings.
   * @param {string} kbSrc - Path to knowledge base JSON.
   * @param {Object} uiConfig - UI element selectors.
   */
  constructor(kbSrc = "kb.json", uiConfig = {}) {
    this.ui = {
      searchInput: uiConfig.searchInput || "#search-input",
      searchButton: uiConfig.searchButton || "#search-btn",
      clearButton: uiConfig.clearButton || "#clear-input",
      resultArea: uiConfig.resultArea || ".results-area",
      chipArea: uiConfig.chipArea || ".suggestion-chips",
      ...uiConfig
    };

    // Generate session ID
    this.sessionId = Date.now().toString(36) + Math.random().toString(36).slice(2);

    // Load kb.json synchronously
    try {
      const xhr = new XMLHttpRequest();
      xhr.open("GET", kbSrc, false);
      xhr.send(null);
      this.kb = JSON.parse(xhr.responseText).facts;
    } catch (e) {
      console.error("Failed to load kb.json:", e);
      this.kb = [];
    }

    // Preprocess knowledge base
    this.df = {};
    this.N = this.kb.length;
    this.totalDocLen = 0;
    this.subtopicIndex = new Map();
    this.kb.forEach((fact, idx) => {
      fact.id = idx;
      fact.views = 0;
      fact.feedback = 0; // Positive/negative feedback
      fact.lastView = Date.now();
      fact.metadata = fact.metadata || {};
      const words = smartTokenize(fact.text + " " + fact.keywords.join(" ") + " " + (fact.metadata.subtopics?.join(" ") || ""));
      fact.docLen = words.length;
      fact.trigrams = words.flatMap(w => generateSubwords(w));
      this.totalDocLen += fact.docLen;
      const subwords = words.flatMap(word => generateSubwords(word));
      const ngramsList = [...ngrams(words, 2)];
      const allTokens = [...words, ...subwords, ...ngramsList];
      fact.bow = bag(allTokens);
      fact.phrases = extractPhrases(words);
      fact.tokens = words;
      fact.embedding = pseudoEmbed(words);
      if (fact.metadata.subtopics) {
        fact.metadata.subtopics.forEach(st => {
          if (!this.subtopicIndex.has(st)) this.subtopicIndex.set(st, []);
          this.subtopicIndex.get(st).push(fact);
        });
      }
      for (const k in fact.bow) this.df[k] = (this.df[k] || 0) + 1;
    });

    // Compute IDF and avgDocLen
    this.idf = {};
    for (const k in this.df) this.idf[k] = Math.log((this.N - this.df[k] + 0.5) / (this.df[k] + 0.5) + 1);
    this.avgDocLen = this.totalDocLen / this.N;

    // Context and memoization
    this.context = {
      currentTopic: null,
      lastTopic: null,
      history: [],
      sessionId: this.sessionId,
      confidence: 0
    };
    this.queryCache = new Map();
    this.debounceTimer = null;
    this.queryPatterns = { definition: 0, comparison: 0, list: 0, general: 0 };

    // Load persisted data
    this.loadPersistedData();
    this.initUI();
  }

  /**
   * Loads fact views, context, and bookmarks from localStorage.
   */
  loadPersistedData() {
    const views = JSON.parse(localStorage.getItem(`nova_fact_views_${this.sessionId}`) || "{}");
    this.kb.forEach(fact => {
      if (views[fact.id]) {
        fact.views = views[fact.id].count;
        fact.lastView = views[fact.id].lastView;
        fact.feedback = views[fact.id].feedback || 0;
        const daysSince = (Date.now() - fact.lastView) / (1000 * 60 * 60 * 24);
        fact.views *= VIEW_DECAY ** daysSince;
        fact.weight = Math.min(1.0, fact.weight + 0.04 * Math.log(1 + fact.views) + 0.02 * fact.feedback);
      }
    });
    this.context.history = JSON.parse(localStorage.getItem(`nova_context_${this.sessionId}`) || "[]");
    this.bookmarks = JSON.parse(localStorage.getItem(`nova_bookmarks_${this.sessionId}`) || "[]");
    this.queryPatterns = JSON.parse(localStorage.getItem(`nova_patterns_${this.sessionId}`) || JSON.stringify(this.queryPatterns));
  }

  /**
   * Saves fact views, context, and bookmarks to localStorage.
   */
  savePersistedData() {
    const views = {};
    this.kb.forEach(fact => {
      if (fact.views > 0 || fact.feedback !== 0) {
        views[fact.id] = { count: fact.views, lastView: fact.lastView, feedback: fact.feedback };
      }
    });
    localStorage.setItem(`nova_fact_views_${this.sessionId}`, JSON.stringify(views));
    localStorage.setItem(`nova_context_${this.sessionId}`, JSON.stringify(this.context.history));
    localStorage.setItem(`nova_bookmarks_${this.sessionId}`, JSON.stringify(this.bookmarks));
    localStorage.setItem(`nova_patterns_${this.sessionId}`, JSON.stringify(this.queryPatterns));
  }

  /**
   * Initializes UI event listeners with debounce.
   */
  initUI() {
    const searchInput = document.querySelector(this.ui.searchInput);
    const searchButton = document.querySelector(this.ui.searchButton);
    const clearButton = document.querySelector(this.ui.clearButton);

    if (searchInput) {
      searchInput.addEventListener("input", () => {
        clearTimeout(this.debounceTimer);
        this.debounceTimer = setTimeout(() => this.handleSearch(), 120);
      });
      searchInput.addEventListener("keypress", e => {
        if (e.key === "Enter") {
          clearTimeout(this.debounceTimer);
          this.handleSearch();
        }
      });
    }

    if (searchButton) {
      searchButton.addEventListener("click", () => {
        clearTimeout(this.debounceTimer);
        this.handleSearch();
      });
    }

    if (clearButton) {
      clearButton.addEventListener("click", () => {
        searchInput.value = "";
        this.renderResults({ topic: "", main: "Ask me anything!", details: [], related: this.bookmarks.slice(0, 8), cites: {} });
        this.renderHistory();
      });
    }

    this.renderHistory();
  }

  /**
   * Handles search input and triggers query processing.
   */
  handleSearch() {
    const searchInput = document.querySelector(this.ui.searchInput);
    const query = searchInput.value.trim();
    if (query) {
      const response = this.processQuery(query);
      this.renderResults(response);
      this.context.history.push({ query, topic: response.topic, topFactId: response.cites[1] });
      if (this.context.history.length > 10) this.context.history.shift();
      this.context.confidence = Math.min(1, query.split(/\s+/).length / 5 + this.context.history.length / 10);
      this.queryPatterns[detectIntent(query)]++;
      this.renderHistory();
      this.saveQuery(query);
      this.savePersistedData();
    }
  }

  /**
   * Saves a query as a bookmark if unique.
   * @param {string} query - Query to save.
   */
  saveQuery(query) {
    if (!this.bookmarks.includes(query)) {
      this.bookmarks.push(query);
      if (this.bookmarks.length > 15) this.bookmarks.shift();
      this.savePersistedData();
    }
  }

  /**
   * Bookmarks a fact.
   * @param {number} factId - Fact ID to bookmark.
   */
  bookmarkFact(factId) {
    const fact = this.kb.find(f => f.id === factId);
    if (fact && !this.bookmarks.includes(fact.text)) {
      this.bookmarks.push(fact.text);
      if (this.bookmarks.length > 15) this.bookmarks.shift();
      this.savePersistedData();
    }
  }

  /**
   * Applies feedback to a fact.
   * @param {number} factId - Fact ID.
   * @param {number} value - Feedback value (+1 or -1).
   */
  applyFeedback(factId, value) {
    const fact = this.kb.find(f => f.id === factId);
    if (fact) {
      fact.feedback += value;
      fact.weight = Math.min(1.0, Math.max(0.7, fact.weight + 0.02 * value));
      this.savePersistedData();
    }
  }

  /**
   * Renders history and bookmarks in the UI.
   */
  renderHistory() {
    const historyArea = document.querySelector(".history-list");
    if (!historyArea) return;

    historyArea.innerHTML = `
      <h3>Recent Queries</h3>
      <ul>
        ${this.context.history.slice(-5).reverse().map(h => `
          <li><button class="history-chip">${detectIntent(h.query).charAt(0).toUpperCase() + detectIntent(h.query).slice(1)}: ${h.query}</button></li>
        `).join("")}
      </ul>
      <h3>Bookmarked</h3>
      <ul>
        ${this.bookmarks.slice(-5).reverse().map(q => `
          <li><button class="history-chip">${q}</button></li>
        `).join("")}
      </ul>
    `;

    document.querySelectorAll(".history-chip").forEach(chip => {
      chip.addEventListener("click", () => {
        const query = chip.textContent.replace(/^(Definition|Comparison|List|General):\s*/i, "");
        document.querySelector(this.ui.searchInput).value = query;
        clearTimeout(this.debounceTimer);
        this.handleSearch();
      });
    });
  }

  /**
   * Processes a user query and returns a structured response.
   * @param {string} query - User's input.
   * @returns {Object} - Response object.
   */
  processQuery(query) {
    if (this.queryCache.has(query)) {
      return this.queryCache.get(query);
    }

    const words = smartTokenize(query);
    const queryPhrases = extractPhrases(words);
    const totalPhrases = queryPhrases.length || 1;
    let qBow = bag(words);

    // Semantic expansion
    const expandedTokens = words.flatMap(w => RELATED_TERMS[w] || []).map(t => ({ term: t, weight: 0.5 }));
    expandedTokens.forEach(({ term, weight }) => {
      qBow[term] = (qBow[term] || 0) + weight;
    });

    // Context blending
    if (words.length < 4 && this.context.history.length > 0) {
      const lastQueries = this.context.history.slice(-3).reverse();
      lastQueries.forEach((h, i) => {
        const weight = [0.4, 0.3, 0.2][i] * this.context.confidence;
        smartTokenize(h.query).forEach(t => {
          qBow[t] = (qBow[t] || 0) + weight;
        });
      });
    }

    // Concept mapping
    const queryConcepts = Object.entries(CONCEPTS)
      .filter(([_, keywords]) => words.some(w => keywords.includes(w)))
      .map(([concept]) => concept);

    const qEmbedding = pseudoEmbed(Object.keys(qBow));
    const intent = detectIntent(query);

    // Rank facts
    const ranked = this.kb.map(fact => {
      // BM25
      let bm25 = 0;
      const k1 = 1.2, b = 0.75;
      for (const term in qBow) {
        if (fact.bow[term]) {
          const tf = fact.bow[term];
          const idf = this.idf[term] || 0;
          const denom = tf + k1 * (1 - b + b * fact.docLen / this.avgDocLen);
          bm25 += idf * tf * (k1 + 1) / denom;
        }
      }

      // Phrase boost
      const matchedPhrases = fact.phrases.filter(p => queryPhrases.includes(p));
      const phraseBoost = matchedPhrases.length / totalPhrases;

      // Fuzzy bonus (weighted trigrams)
      let fuzzyMatches = 0;
      for (const qToken of words) {
        if (fact.tokens.some(t => weightedTrigramSimilarity(qToken, t, this.idf) > 0.5)) {
          fuzzyMatches++;
        }
      }
      const fuzzyBonus = Math.min(0.5, 0.25 * (fuzzyMatches / (words.length || 1)));

      // Dense similarity
      const denseSim = cos(qEmbedding, fact.embedding);

      // Context boost
      const contextBoost = fact.topic === this.context.currentTopic ? 0.2 : 0;

      // Concept boost
      const factConcepts = Object.entries(CONCEPTS)
        .filter(([_, keywords]) => fact.keywords.some(k => keywords.includes(k)))
        .map(([concept]) => concept);
      const conceptBoost = queryConcepts.some(qc => factConcepts.includes(qc)) ? 0.15 : 0;

      // Final score
      const score = (0.45 * bm25 + 0.20 * phraseBoost + 0.15 * fuzzyBonus + 0.10 * denseSim + contextBoost + conceptBoost) * fact.weight;
      return { ...fact, score };
    }).filter(r => r.score > 0.1)
      .sort((a, b) => b.score - a.score);

    if (!ranked.length) {
      const response = {
        topic: "",
        main: `No info found for "${query}". Try something else!`,
        details: [],
        related: this.bookmarks.slice(0, 8),
        cites: {}
      };
      this.queryCache.set(query, response);
      return response;
    }

    // Update fact views and weights
    ranked.slice(0, 8).forEach(fact => {
      fact.views++;
      fact.lastView = Date.now();
      fact.weight = Math.min(1.0, fact.weight + 0.04 * Math.log(1 + fact.views) + 0.02 * fact.feedback);
    });
    this.savePersistedData();

    // Handle intent-specific responses
    const topic = ranked[0].topic;
    let mainFacts = [];
    let supports = [];
    let main = "";
    let cites = {};

    if (intent === "comparison") {
      const entities = query.match(/compare\s+([\w\s-]+)\s+to\s+([\w\s-]+)/i)?.slice(1, 3);
      if (entities) {
        const [e1Facts, e2Facts] = entities.map(e =>
          ranked.filter(f => f.text.toLowerCase().includes(e.toLowerCase()) || f.keywords.some(k => k.includes(e.toLowerCase()))).slice(0, 1)
        );
        mainFacts = [...e1Facts, ...e2Facts].slice(0, 2);
        supports = ranked.filter(f => !mainFacts.includes(f)).slice(0, 2).map(f => f.text);
        ({ main, cites } = this.composeComparison(mainFacts, supports, words, entities));
      }
    } else if (intent === "list") {
      mainFacts = ranked.slice(0, 3);
      supports = ranked.slice(3, 5).map(f => f.text);
      ({ main, cites } = this.composeList(mainFacts, supports, words));
    } else if (intent === "definition") {
      const explainTerm = query.replace(/^(explain|what is|define)\s+/i, "").toLowerCase();
      mainFacts = [ranked.find(f => f.keywords.includes(explainTerm) || f.text.toLowerCase().includes(explainTerm)) || ranked[0]];
      supports = ranked.filter(f => f.id !== mainFacts[0].id && !f.text.toLowerCase().includes(mainFacts[0].text.toLowerCase())).slice(0, 2).map(f => f.text);
      ({ main, cites } = this.composeDefinition(mainFacts[0], supports, words, explainTerm));
    } else {
      mainFacts = [];
      const usedTopics = new Set();
      for (const fact of ranked) {
        if (mainFacts.length < 3) {
          if (usedTopics.size < 2 || usedTopics.has(fact.topic) || mainFacts.length + 1 === ranked.length) {
            mainFacts.push(fact);
            usedTopics.add(fact.topic);
          }
        } else {
          break;
        }
      }
      supports = ranked.filter(f => !mainFacts.includes(f)).slice(0, 2).map(f => f.text);
      ({ main, cites } = this.composeGeneral(mainFacts, supports, words));
    }

    const details = ranked.filter(f => !mainFacts.includes(f)).slice(0, 5).map(f => this.highlightTokens(f.text, words));
    const related = this.generateChips(query, ranked, topic, intent);

    // Update context
    this.context.lastTopic = this.context.currentTopic;
    this.context.currentTopic = topic;

    const response = { topic, main, details, related, cites };
    this.queryCache.set(query, response);
    return response;
  }

  /**
   * Composes a comparison response.
   * @param {Array} facts - Main facts for comparison.
   * @param {string[]} supports - Supporting texts.
   * @param {string[]} queryTokens - Query tokens for highlighting.
   * @param {string[]} entities - Entities being compared.
   * @returns {Object} - { main: string, cites: Object }
   */
  composeComparison(facts, supports, queryTokens, entities) {
    const template = responseTemplates.comparison;
    const highlight = text => this.highlightTokens(text, queryTokens);
    const cites = {};
    let main = "No comparison available.";
    if (facts.length >= 1) {
      cites[1] = facts[0].id;
      if (facts.length === 2) {
        cites[2] = facts[1].id;
        main = template.structure(facts[0], facts[1]);
      } else {
        main = template.structure(facts[0], null);
      }
      main = highlight(main);
      let wordCount = main.split(/\s+/).length;
      supports.forEach((s, i) => {
        if (wordCount < 100) {
          const snippet = s.split(/\s+/).slice(0, Math.max(1, 50 - wordCount / 2)).join(" ");
          wordCount += snippet.split(/\s+/).length;
          cites[i + 3] = facts[0].id + i + 1;
          main += ` ${template.support(highlight(snippet), i + 3)}`;
        }
      });
    }
    const words = main.split(/\s+/).slice(0, 100).join(" ");
    return { main: words + (words.endsWith(".") ? "" : "..."), cites };
  }

  /**
   * Composes a list response.
   * @param {Array} facts - Main facts.
   * @param {string[]} supports - Supporting texts.
   * @param {string[]} queryTokens - Query tokens for highlighting.
   * @returns {Object} - { main: string, cites: Object }
   */
  composeList(facts, supports, queryTokens) {
    const template = responseTemplates.list;
    const highlight = text => this.highlightTokens(text, queryTokens);
    const cites = {};
    const main = template.structure(facts.map(f => ({ ...f, text: highlight(f.text) })));
    facts.forEach((f, i) => cites[i + 1] = f.id);
    let wordCount = main.split(/\s+/).length;
    supports.forEach((s, i) => {
      if (wordCount < 100) {
        const snippet = s.split(/\s+/).slice(0, Math.max(1, 50 - wordCount / 2)).join(" ");
        wordCount += snippet.split(/\s+/).length;
        cites[i + facts.length + 1] = f.id + i + 1;
        main += ` ${template.support(highlight(snippet), i + facts.length + 1)}`;
      }
    });
    const words = main.split(/\s+/).slice(0, 100).join(" ");
    return { main: words + (words.endsWith(".") ? "" : "..."), cites };
  }

  /**
   * Composes a definition response.
   * @param {Object} mainFact - Main fact.
   * @param {string[]} supports - Supporting texts.
   * @param {string[]} queryTokens - Query tokens for highlighting.
   * @param {string} explainTerm - Term to define.
   * @returns {Object} - { main: string, cites: Object }
   */
  composeDefinition(mainFact, supports, queryTokens, explainTerm) {
    const template = responseTemplates.definition;
    const highlight = text => this.highlightTokens(text, queryTokens);
    const cites = { 1: mainFact.id };
    let main = highlight(mainFact.text);
    if (!mainFact.text.toLowerCase().includes(explainTerm) && !mainFact.keywords.includes(explainTerm)) {
      const related = RELATED_TERMS[explainTerm] || [];
      main = `${explainTerm.charAt(0).toUpperCase() + explainTerm.slice(1)} is a ${related.join(" or ")} technology.`;
    }
    main = template.structure({ text: main });
    let wordCount = main.split(/\s+/).length;
    supports.forEach((s, i) => {
      if (wordCount < 100) {
        const snippet = s.split(/\s+/).slice(0, Math.max(1, 50 - wordCount / 2)).join(" ");
        wordCount += snippet.split(/\s+/).length;
        cites[i + 2] = mainFact.id + i + 1;
        main += ` ${template.support(highlight(snippet), i + 2)}`;
      }
    });
    const words = main.split(/\s+/).slice(0, 100).join(" ");
    return { main: words + (words.endsWith(".") ? "" : "..."), cites };
  }

  /**
   * Composes a general response.
   * @param {Array} facts - Main facts.
   * @param {string[]} supports - Supporting texts.
   * @param {string[]} queryTokens - Query tokens for highlighting.
   * @returns {Object} - { main: string, cites: Object }
   */
  composeGeneral(facts, supports, queryTokens) {
    const template = responseTemplates.general;
    const highlight = text => this.highlightTokens(text, queryTokens);
    const cites = {};
    let main = template.structure(facts.map(f => ({ ...f, text: highlight(f.text) })));
    facts.forEach((f, i) => cites[i + 1] = f.id);
    let wordCount = main.split(/\s+/).length;
    supports.forEach((s, i) => {
      if (wordCount < 100) {
        const snippet = s.split(/\s+/).slice(0, Math.max(1, 50 - wordCount / 2)).join(" ");
        wordCount += snippet.split(/\s+/).length;
        cites[i + facts.length + 1] = facts[0].id + i + 1;
        main += ` ${template.support(highlight(snippet), i + facts.length + 1)}`;
      }
    });
    const words = main.split(/\s+/).slice(0, 100).join(" ");
    return { main: words + (words.endsWith(".") ? "" : "..."), cites };
  }

  /**
   * Highlights query tokens in text.
   * @param {string} text - Input text.
   * @param {string[]} queryTokens - Tokens to highlight.
   * @returns {string} - Text with highlighted tokens.
   */
  highlightTokens(text, queryTokens) {
    let result = text;
    for (const token of queryTokens) {
      const regex = new RegExp(`\\b${token}\\b`, "gi");
      result = result.replace(regex, `<mark>$&</mark>`);
    }
    return result;
  }

  /**
   * Generates follow-up query chips.
   * @param {string} query - Original query.
   * @param {Array} ranked - Ranked facts.
   * @param {string} topic - Current topic.
   * @param {string} intent - Detected intent.
   * @returns {Array} - Chip strings.
   */
  generateChips(query, ranked, topic, intent) {
    const patternWeights = Object.values(this.queryPatterns);
    const totalPatterns = patternWeights.reduce((s, x) => s + x, 1);
    const intentBias = this.queryPatterns[intent] / totalPatterns;

    const semanticChips = ranked.slice(3, 10).map(f => {
      const snippet = f.text.split(" ").slice(0, 5).join(" ") + "...";
      return snippet.length <= 50 ? snippet : null;
    }).filter(Boolean).slice(0, 2);

    const deepDiveChips = smartTokenize(query).slice(0, intentBias > 0.5 ? 4 : 2).map(t => {
      const chip = `How does ${t} work?`;
      return chip.length <= 50 ? chip : null;
    }).filter(Boolean);

    const stockChip = [`Future of ${topic}`].filter(c => c.length <= 50);
    const comparisonChip = this.context.lastTopic && this.context.lastTopic !== topic 
      ? [`Compare ${topic} to ${this.context.lastTopic}`].filter(c => c.length <= 50)
      : [];

    const conceptChips = Object.keys(CONCEPTS)
      .filter(c => CONCEPTS[c].some(k => ranked[0].keywords.includes(k)))
      .map(c => `Explore ${c.replace("-", " ")}`)
      .filter(c => c.length <= 50)
      .slice(0, 2);

    const trendingTopic = Object.entries(this.queryPatterns)
      .filter(([k]) => k !== intent)
      .sort((a, b) => b[1] - a[1])[0]?.[0];
    const trendingChip = trendingTopic ? [`Try a ${trendingTopic} query`].filter(c => c.length <= 50) : [];

    let allChips = [...semanticChips, ...deepDiveChips, ...stockChip, ...comparisonChip, ...conceptChips, ...trendingChip];
    allChips = shuffle(uniq(allChips));

    let idx = allChips.length + 3;
    while (allChips.length < 8 && idx < ranked.length) {
      const extra = ranked[idx].text.split(" ").slice(0, 5).join(" ") + "...";
      if (extra.length <= 50 && !allChips.includes(extra)) allChips.push(extra);
      idx++;
    }

    return allChips.slice(0, 8);
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
      <div class="result-card">
        <h2>${response.topic || "General"}</h2>
        <p>${response.main}</p>
        ${response.details.length ? `
          <h3>More Details</h3>
          <ul>
            ${response.details.map(d => `<li>${d} <button class="feedback-up" data-id="${response.cites[1]}">👍</button><button class="feedback-down" data-id="${response.cites[1]}">👎</button></li>`).join("")}
          </ul>
        ` : ""}
      </div>
    `;

    chipArea.innerHTML = response.related.map(chip => `
      <button class="suggestion-chip">${chip}</button>
    `).join("");

    document.querySelectorAll(".suggestion-chip").forEach(chip => {
      chip.addEventListener("click", () => {
        document.querySelector(this.ui.searchInput).value = chip.textContent.replace("?", "");
        clearTimeout(this.debounceTimer);
        this.handleSearch();
      });
    });

    document.querySelectorAll(".feedback-up").forEach(btn => {
      btn.addEventListener("click", () => {
        this.applyFeedback(parseInt(btn.dataset.id), 1);
        btn.disabled = true;
      });
    });

    document.querySelectorAll(".feedback-down").forEach(btn => {
      btn.addEventListener("click", () => {
        this.applyFeedback(parseInt(btn.dataset.id), -1);
        btn.disabled = true;
      });
    });
  }
}

/* ---------- Export Class ---------- */
global.NovaAI = NovaAI;

// Auto-initialize on load
document.addEventListener("DOMContentLoaded", () => {
  new NovaAI("kb.json");
});

})(typeof window !== "undefined" ? window : this);