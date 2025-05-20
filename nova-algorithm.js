(function (global) {

const STOP_WORDS = new Set(["a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "with", "what", "which", "when", "where", "how", "why"]);
const SYNONYMS = { ai: "artificial intelligence", llm: "large language model", vr: "virtual reality", btc: "bitcoin" };
const RELATED_TERMS = { ai: ["machine-learning", "neural-networks", "deep-learning"], blockchain: ["decentralized", "ledger", "cryptocurrency"], quantum: ["qubits", "superposition", "entanglement"], health: ["wellness", "nutrition", "fitness"], environment: ["sustainability", "climate", "ecology"] };
const CONCEPTS = { "future-tech": ["ai", "quantum-tech", "blockchain", "digital-twins", "augmented-reality"], "green-tech": ["environment", "biochar", "carbon-capture", "offshore-wind"], "data-security": ["zero-trust", "cryptography", "blockchain"], "human-augmentation": ["health", "wearable-tech", "augmented-reality", "soft-robotics"] };
const STEM_SUFFIXES = ["ing", "ed", "es", "s"];
const EMBED_DIM = 100;
const VIEW_DECAY = 0.95;
const MAX_CACHE_SIZE = 1000;

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

const stemWord = word => {
  for (const suffix of STEM_SUFFIXES) {
    if (word.endsWith(suffix)) return word.slice(0, -suffix.length);
  }
  return word;
};

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
  if (tokenCache.size > MAX_CACHE_SIZE) tokenCache.delete(tokenCache.keys().next().value);
  return result;
};

const generateSubwords = word => {
  if (word.length <= 3) return [];
  const subwords = [];
  for (let i = 0; i <= word.length - 3; i++) subwords.push(word.slice(i, i + 3));
  return subwords;
};

const ngrams = (arr, n) => {
  const out = [];
  for (let i = 0; i <= arr.length - n; i++) out.push(arr.slice(i, i + n).join(" "));
  return out;
};

const extractPhrases = tokens => [...ngrams(tokens, 2), ...ngrams(tokens, 3)];

const bag = arr => arr.reduce((obj, word) => {
  obj[word] = (obj[word] || 0) + 1;
  return obj;
}, {});

const simpleHash = str => {
  let hash = 0;
  for (let i = 0; i < str.length; i++) hash = (hash * 31 + str.charCodeAt(i)) & 0x7fffffff;
  return hash;
};

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

const cos = (a, b) => {
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] ** 2;
    magB += b[i] ** 2;
  }
  return dot ? dot / Math.sqrt(magA * magB) : 0;
};

const uniq = arr => [...new Set(arr)];

const shuffle = arr => arr.slice().sort(() => 0.5 - Math.random());

// Updated intent patterns to better differentiate query types
const INTENT_PATTERNS = {
  definition: /^(explain|what is|define)\s+/i,
  comparison: /compare.*to/i,
  list: /(list|top|best).*of/i,
  weather: /(weather|forecast)\s*(in\s+[\w\s]+)?$/i,
  food: /(food|restaurants)\s+near\s+me/i,
  address: /^(search|address)\s+[\w\s,.-]+$/i,
  general: /.*/  // Fallback for regular questions
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
    connectors: ["Similarly", "In addition"], 
    structure: (facts, highlightFn) => {
      const highlightedFacts = facts.map(f => ({ ...f, text: highlightFn(f.text) }));
      if (highlightedFacts.length === 1) return highlightedFacts[0].text + `<sup>[1]</sup>`;
      let response = highlightedFacts[0].text + `<sup>[1]</sup>`;
      for (let i = 1; i < highlightedFacts.length; i++) {
        const sim = cos(highlightedFacts[i - 1].embedding, highlightedFacts[i].embedding);
        const connector = sim > 0.7 ? "Similarly," : "In addition,";
        response += ` ${connector} ${highlightedFacts[i].text.toLowerCase()}<sup>[${i + 1}]</sup>`;
      }
      return response;
    }, 
    support: (text, idx) => `${text.toLowerCase()}<sup>[${idx}]</sup>`
  }
};

const detectIntent = query => {
  for (const [intent, pattern] of Object.entries(INTENT_PATTERNS)) {
    if (pattern.test(query)) return intent;
  }
  return "general";
};

class NovaAI {
  constructor(kbSrc = "kb.json", uiConfig = {}) {
    this.ui = {
      searchInput: uiConfig.searchInput || "#searchInput",
      resultArea: uiConfig.resultArea || "#resultArea",
      chipArea: uiConfig.chipArea || "#chipArea",
      historyArea: uiConfig.historyArea || "#historyArea",
      ...uiConfig
    };

    this.sessionId = Date.now().toString(36) + Math.random().toString(36).slice(2);

    try {
      const xhr = new XMLHttpRequest();
      xhr.open("GET", kbSrc, false);
      xhr.send(null);
      this.kb = JSON.parse(xhr.responseText).facts;
    } catch (e) {
      console.error("Failed to load kb.json:", e);
      this.kb = [];
    }

    this.df = {};
    this.N = this.kb.length;
    this.totalDocLen = 0;
    this.subtopicIndex = new Map();
    this.categoryIndex = new Map();
    this.kb.forEach((fact, idx) => {
      fact.id = fact.id || idx;
      fact.views = 0;
      fact.feedback = 0;
      fact.lastView = Date.now();
      fact.metadata = fact.metadata || {};
      fact.category = fact.metadata.category || fact.topic.split('/')[0];
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
      fact.embedding = fact.embedding || pseudoEmbed(words);
      if (fact.metadata.subtopics) {
        fact.metadata.subtopics.forEach(st => {
          if (!this.subtopicIndex.has(st)) this.subtopicIndex.set(st, []);
          this.subtopicIndex.get(st).push(fact);
        });
      }
      if (fact.category) {
        if (!this.categoryIndex.has(fact.category)) this.categoryIndex.set(fact.category, []);
        this.categoryIndex.get(fact.category).push(fact);
      }
      for (const k in fact.bow) this.df[k] = (this.df[k] || 0) + 1;
    });

    this.idf = {};
    for (const k in this.df) this.idf[k] = Math.log((this.N - this.df[k] + 0.5) / (this.df[k] + 0.5) + 1);
    this.avgDocLen = this.totalDocLen / this.N;

    this.context = { currentTopic: null, lastTopic: null, currentCategory: null, history: [], sessionId: this.sessionId, confidence: 0 };
    this.queryCache = new Map();
    this.debounceTimer = null;
    this.queryPatterns = { definition: 0, comparison: 0, list: 0, weather: 0, food: 0, address: 0, general: 0 };

    this.loadPersistedData();
    this.initUI();
  }

  loadPersistedData() {
    const views = JSON.parse(localStorage.getItem(`nova_fact_views_${this.sessionId}`) || "{}");
    this.kb.forEach(fact => {
      if (views[fact.id]) {
        fact.views = views[fact.id].count;
        fact.lastView = views[fact.id].lastView;
        fact.feedback = views[fact.id].feedback || 0;
        const daysSince = (Date.now() - fact.lastView) / (1000 * 60 * 60 * 24);
        fact.views *= VIEW_DECAY ** daysSince;
        fact.weight = Math.min(1.0, (fact.metadata.priority || 0.8) + 0.04 * Math.log(1 + fact.views) + 0.02 * fact.feedback);
      } else {
        fact.weight = fact.metadata.priority || 0.8;
      }
    });
    this.context.history = JSON.parse(localStorage.getItem(`nova_context_${this.sessionId}`) || "[]");
    this.bookmarks = JSON.parse(localStorage.getItem(`nova_bookmarks_${this.sessionId}`) || "[]");
    this.queryPatterns = JSON.parse(localStorage.getItem(`nova_patterns_${this.sessionId}`) || JSON.stringify(this.queryPatterns));
  }

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

  initUI() {
    const searchInput = document.querySelector(this.ui.searchInput);
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

    this.renderHistory();
    // Initial render with examples for all supported query types
    this.renderResults({ 
      category: "General", 
      main: "Ask me anything! Try 'weather', 'food near me', 'search 123 Main St', or 'what is AI'.", 
      details: [], 
      related: this.bookmarks.slice(0, 8), 
      cites: {} 
    });
  }

  handleSearch() {
    const searchInput = document.querySelector(this.ui.searchInput);
    const query = searchInput.value.trim();
    if (!query) return;

    const intent = detectIntent(query);
    this.queryPatterns[intent] = (this.queryPatterns[intent] || 0) + 1;

    // Send weather, food, and address queries to the Flask backend
    if (["weather", "food", "address"].includes(intent)) {
      fetch('http://localhost:5000/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      })
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        this.context.currentTopic = data.topic || "";
        this.context.currentCategory = data.category || "";
        this.context.confidence = 0.9; // High confidence for API results
        this.renderResults(data);
      })
      .catch(error => {
        console.error('Error:', error);
        this.renderResults({ 
          category: "Error", 
          main: "Failed to fetch API data. Is the backend running at http://localhost:5000?", 
          details: [], 
          related: [], 
          cites: {} 
        });
      });
    } else {
      // Process regular questions (definition, comparison, list, general) using kb.json
      const response = this.processQuery(query);
      this.context.currentTopic = response.topic || "";
      this.context.currentCategory = response.category || "";
      this.context.confidence = response.score ? 0.6 : 0.3;
      this.renderResults(response);
    }

    this.context.history.push({ query, topic: this.context.currentTopic, topFactId: null });
    if (this.context.history.length > 10) this.context.history.shift();
    this.renderHistory();
    this.savePersistedData();
  }

  saveQuery(query) {
    if (!this.bookmarks.includes(query)) {
      this.bookmarks.push(query);
      if (this.bookmarks.length > 15) this.bookmarks.shift();
      this.savePersistedData();
    }
  }

  bookmarkFact(factId) {
    const fact = this.kb.find(f => f.id === factId);
    if (fact && !this.bookmarks.includes(fact.text)) {
      this.bookmarks.push(fact.text);
      if (this.bookmarks.length > 15) this.bookmarks.shift();
      this.savePersistedData();
    }
  }

  applyFeedback(factId, value) {
    const fact = this.kb.find(f => f.id === factId);
    if (fact) {
      fact.feedback += value;
      fact.weight = Math.min(1.0, Math.max(0.7, fact.weight + 0.02 * value));
      this.savePersistedData();
    }
  }

  renderHistory() {
    const historyArea = document.querySelector(this.ui.historyArea);
    if (!historyArea) return;

    historyArea.innerHTML = `
      <div class="history-section">
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
      </div>
    `;

    document.querySelectorAll(".history-chip").forEach(chip => {
      chip.addEventListener("click", () => {
        const query = chip.textContent.replace(/^(Definition|Comparison|List|Weather|Food|Address|General):\s*/i, "");
        document.querySelector(this.ui.searchInput).value = query;
        clearTimeout(this.debounceTimer);
        this.handleSearch();
      });
    });
  }

  processQuery(query) {
    if (this.queryCache.has(query)) return this.queryCache.get(query);

    const words = smartTokenize(query);
    const queryPhrases = extractPhrases(words);
    const totalPhrases = queryPhrases.length || 1;
    let qBow = bag(words);

    const expandedTokens = words.flatMap(w => RELATED_TERMS[w] || []).map(t => ({ term: t, weight: 0.5 }));
    expandedTokens.forEach(({ term, weight }) => {
      qBow[term] = (qBow[term] || 0) + weight;
    });

    if (words.length < 4 && this.context.history.length > 0) {
      const lastQueries = this.context.history.slice(-3).reverse();
      lastQueries.forEach((h, i) => {
        const weight = [0.4, 0.3, 0.2][i] * this.context.confidence;
        smartTokenize(h.query).forEach(t => {
          qBow[t] = (qBow[t] || 0) + weight;
        });
      });
    }

    const queryConcepts = Object.entries(CONCEPTS)
      .filter(([_, keywords]) => words.some(w => keywords.includes(w)))
      .map(([concept]) => concept);

    const querySubtopics = words.filter(w => this.subtopicIndex.has(w));
    const queryCategories = words.filter(w => this.categoryIndex.has(w));

    const qEmbedding = pseudoEmbed(Object.keys(qBow));
    const intent = detectIntent(query);

    const userConceptProfile = {};
    this.context.history.forEach(h => {
      const hWords = smartTokenize(h.query);
      const hConcepts = Object.entries(CONCEPTS)
        .filter(([_, keywords]) => hWords.some(w => keywords.includes(w)))
        .map(([concept]) => concept);
      hConcepts.forEach(c => {
        userConceptProfile[c] = (userConceptProfile[c] || 0) + 1;
      });
    });
    const numQueries = this.context.history.length;

    const ranked = this.kb.map(fact => {
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

      const matchedPhrases = fact.phrases.filter(p => queryPhrases.includes(p));
      const phraseBoost = matchedPhrases.length / totalPhrases;

      let fuzzyMatches = 0;
      for (const qToken of words) {
        if (fact.tokens.some(t => weightedTrigramSimilarity(qToken, t, this.idf) > 0.5)) fuzzyMatches++;
      }
      const fuzzyBonus = Math.min(0.5, 0.25 * (fuzzyMatches / (words.length || 1)));

      const denseSim = cos(qEmbedding, fact.embedding);
      const contextBoost = fact.topic === this.context.currentTopic ? 0.2 : 0;
      const categoryBoost = fact.category === this.context.currentCategory ? 0.15 : 0;

      const factConcepts = Object.entries(CONCEPTS)
        .filter(([_, keywords]) => fact.keywords.some(k => keywords.includes(k)))
        .map(([concept]) => concept);
      const conceptBoost = queryConcepts.some(qc => factConcepts.includes(qc)) ? 0.15 : 0;

      const personalizationScore = factConcepts.reduce((sum, c) => sum + (userConceptProfile[c] || 0), 0) / (numQueries || 1);
      const freshnessScore = fact.metadata.updatedAt
        ? Math.max(0, 1 - (Date.now() - new Date(fact.metadata.updatedAt).getTime()) / (1000 * 60 * 60 * 24 * 365))
        : 0;
      const subtopicBoost = fact.metadata.subtopics?.some(st => querySubtopics.includes(st)) ? 0.1 : 0;
      const categoryMatchBoost = queryCategories.includes(fact.category) ? 0.2 : 0;

      const score = (0.40 * bm25 + 0.20 * phraseBoost + 0.15 * fuzzyBonus + 0.10 * denseSim + contextBoost + categoryBoost + conceptBoost + 0.05 * personalizationScore + 0.05 * freshnessScore + subtopicBoost + categoryMatchBoost) * fact.weight;
      return { ...fact, score };
    }).filter(r => r.score > 0.1)
      .sort((a, b) => b.score - a.score);

    if (!ranked.length) {
      const response = { topic: "", category: "General", main: `No info found for "${query}". Try something else!`, details: [], related: this.bookmarks.slice(0, 8), cites: {} };
      this.queryCache.set(query, response);
      return response;
    }

    ranked.slice(0, 8).forEach(fact => {
      fact.views++;
      fact.lastView = Date.now();
      fact.weight = Math.min(1.0, fact.weight + 0.04 * Math.log(1 + fact.views) + 0.02 * fact.feedback);
    });
    this.savePersistedData();

    const topic = ranked[0].topic;
    const category = ranked[0].category;
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
        } else break;
      }
      supports = ranked.filter(f => !mainFacts.includes(f)).slice(0, 2).map(f => f.text);
      ({ main, cites } = this.composeGeneral(mainFacts, supports, words));
    }

    const details = ranked.filter(f => !mainFacts.includes(f)).slice(0, 5).map(f => this.highlightTokens(f.text, words));
    const related = this.generateChips(query, ranked, topic, category, intent);

    const response = { topic, category, main, details, related, cites, score: ranked[0].score };
    this.queryCache.set(query, response);
    return response;
  }

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
        cites[i + facts.length + 1] = facts[0].id + i + 1;
        main += ` ${template.support(highlight(snippet), i + facts.length + 1)}`;
      }
    });
    const words = main.split(/\s+/).slice(0, 100).join(" ");
    return { main: words + (words.endsWith(".") ? "" : "..."), cites };
  }

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

  composeGeneral(facts, supports, queryTokens) {
    const template = responseTemplates.general;
    const highlight = text => this.highlightTokens(text, queryTokens);
    const cites = {};
    let main = template.structure(facts, highlight);
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

  highlightTokens(text, queryTokens) {
    let result = text;
    for (const token of queryTokens) {
      const regex = new RegExp(`\\b${token}\\b`, "gi");
      result = result.replace(regex, `<mark>$&</mark>`);
    }
    return result;
  }

  generateChips(query, ranked, topic, category, intent) {
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

    const categoryChip = this.categoryIndex.has(category)
      ? [`More in ${category}`].filter(c => c.length <= 50)
      : [];

    const relatedTopics = ranked[0].relations?.relatedTopics || [];
    const relatedTopicChips = relatedTopics.map(t => `Explore ${t}`).filter(c => c.length <= 50).slice(0, 2);

    const trendingTopic = Object.entries(this.queryPatterns)
      .filter(([k]) => k !== intent)
      .sort((a, b) => b[1] - a[1])[0]?.[0];
    const trendingChip = trendingTopic ? [`Try a ${trendingTopic} query`] : [];

    let allChips = [...semanticChips, ...deepDiveChips, ...stockChip, ...comparisonChip, ...conceptChips, ...categoryChip, ...relatedTopicChips, ...trendingChip];
    allChips = shuffle(uniq(allChips));

    let idx = allChips.length + 3;
    while (allChips.length < 8 && idx < ranked.length) {
      const extra = ranked[idx].text.split(" ").slice(0, 5).join(" ") + "...";
      if (extra.length <= 50 && !allChips.includes(extra)) allChips.push(extra);
      idx++;
    }

    return allChips.slice(0, 8);
  }

 renderResults(response) {
  const resultArea = document.querySelector(this.ui.resultArea);
  const chipArea = document.querySelector(this.ui.chipArea);
  if (!resultArea || !chipArea) return;

  resultArea.innerHTML = `
    <h2>${response.category || "General"}</h2>
    <p>${response.main || "No results"}</p>
    ${response.details && response.details.length ? `
      <h3>More Details</h3>
      <ul>
        ${response.details.map((d, i) => `<li>${d} ${response.cites && response.cites[i + 1] ? `<button class="feedback-up" data-id="${response.cites[i + 1]}">👍</button><button class="feedback-down" data-id="${response.cites[i + 1]}">👎</button>` : ""}</li>`).join("")}
      </ul>
    ` : ""}
    ${response.links && response.links.length ? `<h3>Links</h3><ul>${response.links.map(l => `<li><a href="${l}" target="_blank">${l}</a></li>`).join("")}</ul>` : ""}
  `;

  chipArea.innerHTML = response.related?.map(chip => `
    <button class="suggestion-chip">${chip}</button>
  `).join("") || "";

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

global.NovaAI = NovaAI;

document.addEventListener("DOMContentLoaded", () => {
  new NovaAI("kb.json", {
    searchInput: "#searchInput",
    resultArea: "#resultArea",
    chipArea: "#chipArea",
    historyArea: "#historyArea"
  });
});

})(typeof window !== "undefined" ? window : this);