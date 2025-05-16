/* ======================================================================
   nova-algorithm.js  ·  v3: advanced NLP, context-aware ranking, natural responses
   ====================================================================== */

(function (global) {

/* ---------- Text Utilities ---------- */
const STOP = new Set("a an and are as at be by for from has have in is it of on or that the this to with what which when where how why".split(" "));
const ALIAS = {
  ai: "artificial intelligence", llm: "large language model", ml: "machine learning",
  iot: "internet of things", nlp: "natural language processing", vr: "virtual reality",
  ar: "augmented reality", crypto: "blockchain", wasm: "webassembly", cbt: "cognitive behavioral therapy"
};
const SYNONYMS = {
  blockchain: ["crypto", "distributed ledger"], ai: ["machine learning", "neural network"],
  quantum: ["qubit", "superposition"], climate: ["weather", "environment"],
  coding: ["programming", "software"], ethics: ["morality", "philosophy"]
};
const SIMPLE_LEMMA = {
  running: "run", runs: "run", ran: "run", coded: "code", coding: "code",
  codes: "code", technologies: "technology", models: "model", facts: "fact"
};

const expand = w => ALIAS[w] ? ALIAS[w].split(" ") : [w];
const lemmatize = w => SIMPLE_LEMMA[w] || w.replace(/s$/, "");
const tokens = txt => {
  const out = [];
  txt.toLowerCase().split(/[^a-z0-9]+/).forEach(w => {
    if (!w || STOP.has(w)) return;
    const lemma = lemmatize(w);
    out.push(...expand(lemma));
    if (SYNONYMS[lemma]) out.push(...SYNONYMS[lemma].map(lemmatize));
  });
  return out;
};
const bag = a => a.reduce((o, w) => (o[w] = (o[w] || 0) + 1, o), {});
const ngrams = (a, n) => {
  const out = [];
  for (let i = 0; i <= a.length - n; i++) out.push(a.slice(i, i + n).join("_"));
  return out;
};
const cos = (a, b) => {
  let d = 0, ma = 0, mb = 0;
  for (const k in a) { ma += a[k] ** 2; if (b[k]) d += a[k] * b[k]; }
  for (const k in b) mb += b[k] ** 2;
  return d ? d / Math.sqrt(ma * mb) : 0;
};
const uniq = a => [...new Set(a)];
const shuffle = a => a.slice().sort(() => 0.5 - Math.random());

/* ---------- NovaAI ---------- */
class NovaAI {
  constructor(src = "kb.json") {
    // Sync-load knowledge base
    const x = new XMLHttpRequest();
    x.open("GET", src, false);
    x.send(null);
    this.kb = JSON.parse(x.responseText).facts;

    // Preprocess facts: TF-IDF vectors, n-grams, and metadata
    this.df = {};
    this.N = this.kb.length;
    this.kb.forEach(f => {
      f.toks = tokens(f.text);
      f.bow = bag([...f.toks, ...tokens(f.keywords.join(" "))]);
      f.ngrams = [...ngrams(f.toks, 2), ...ngrams(f.toks, 3)];
      f.ngramBow = bag(f.ngrams);
      for (const k in f.bow) this.df[k] = (this.df[k] || 0) + 1;
      for (const k in f.ngramBow) this.df[k] = (this.df[k] || 0) + 1;
    });

    this.idf = {};
    for (const k in this.df) this.idf[k] = Math.log(this.N / (this.df[k] || 1));
    this.kb.forEach(f => {
      const v = {};
      for (const k in f.bow) v[k] = f.bow[k] * this.idf[k];
      for (const k in f.ngramBow) v[k] = f.ngramBow[k] * (this.idf[k] || 1);
      const n = Math.sqrt(Object.values(v).reduce((s, x) => s + x * x, 0)) || 1;
      for (const k in v) v[k] /= n;
      f.vec = v;
    });

    // Cache topic and subtopic lists
    this.topics = uniq(this.kb.map(f => f.topic));
    this.subtopics = uniq(this.kb.flatMap(f => f.subtopics));

    // Context tracking
    this.context = {
      currentTopic: null,
      lastTopic: null,
      history: [],
      maxHistory: 20
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
      }
    };

    // Sentiment phrases
    this.sentiments = {
      positive: ["Great question!", "Love the enthusiasm!", "Awesome topic!"],
      curious: ["That's a deep one!", "Curious angle, let's dive in!", "Intriguing query!"],
      neutral: ["Got it, let's explore.", "Here's the scoop.", "Alright, let's tackle it."],
      confused: ["Let's clear that up!", "I hear you, let's sort it out.", "No worries, I'll clarify."]
    };
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
    const ranked = this.kb.map(f => {
      const sim = cos(qVec, f.vec);
      const topicBoost = this.context.currentTopic === f.topic ? 1.2 : 1;
      const historyBoost = this.context.history.slice(-3).some(h => h.topic === f.topic) ? 1.1 : 1;
      return { ...f, score: sim * f.weight * topicBoost * historyBoost };
    }).filter(r => r.score > 0.1)
      .sort((a, b) => b.score - a.score);

    if (!ranked.length) return this.sorry(q);

    // Compose response
    const topic = ranked[0].topic;
    const topFacts = ranked.slice(0, 3);
    const main = this.composeResponse(topFacts, intent, sentiment, entities, topic, q);
    const details = ranked.slice(3, 6).map(f => f.text);
    const related = this.generateChips(topic, entities, q, ranked);

    // Update context
    this.context.lastTopic = this.context.currentTopic;
    this.context.currentTopic = topic;
    this.context.history.push({ query: q, topic, entities, intent, sentiment, ts: Date.now() });
    if (this.context.history.length > this.context.maxHistory) this.context.history.shift();

    return { topic, main, details, stats: null, quote: null, related };
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
    const n = Math.sqrt(Object.values(qVec).reduce((s, x) => s + x * x, 0)) || 1;
    for (const k in qVec) qVec[k] /= n;

    // Intent detection
    const lowerQ = query.toLowerCase();
    let intent = "description";
    const isAmbiguous = lowerQ.length < 5 || ["it", "this", "that"].includes(lowerQ);
    const intentPatterns = {
      question: /^(what|how|why|when|where|who)\b|.*\?$/,
      explanation: /\b(how|works|process|explain|mechanism)\b/,
      description: /\b(what|about|describe|is|tell)\b/,
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
      positive: ["great", "awesome", "excited", "love"],
      curious: ["wonder", "curious", "why", "how"],
      confused: ["confused", "unclear", "not sure"]
    };
    for (const [key, keywords] of Object.entries(sentimentKeywords)) {
      if (keywords.some(k => lowerQ.includes(k))) {
        sentiment = key;
        break;
      }
    }

    // Entity extraction
    const entities = [];
    this.kb.forEach(f => {
      f.keywords.forEach(k => {
        if (lowerQ.includes(k) && !entities.some(e => e.keyword === k)) {
          entities.push({ topic: f.topic, keyword: k, weight: k.length / lowerQ.length });
        }
      });
    });
    if (this.context.history.length) {
      const recentEntities = this.context.history.slice(-3).flatMap(h => h.entities || []);
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
    const isFollowUp = this.context.history.some(h => h.topic === topic);
    const topicShift = this.context.lastTopic && this.context.lastTopic !== topic;

    let response = "";
    if (topicShift) response += `Switching to ${topic}. `;
    else if (isFollowUp) response += `More on ${topic}. `;
    
    response += this.sentiments[sentiment][Math.floor(Math.random() * 3)] + " ";

    const content = facts.length > 1 ? 
      `${facts[0].text} Also, ${facts[1].text}${facts[2] ? `. Plus, ${facts[2].text}` : ""}` : 
      facts[0].text;

    response += template.main
      .replace("{sentiment}", "")
      .replace("{entity}", entity)
      .replace("{topic}", topic)
      .replace("{content}", content)
      .replace("{mechanism}", facts[0].text.split(".")[0] || "key processes")
      .replace("{impact}", "its field")
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
    const semantic = ranked.slice(3, 8).map(f => {
      const t = f.text.split(".")[0];
      return t.length > 50 ? t.slice(0, 47) + "…" : t;
    });
    const refine = uniq(tokens(query)).slice(0, 3).map(w => `More on ${w}?`);
    const stock = [
      `What’s new in ${topic}?`,
      `How does ${topic} work?`,
      `Key trends in ${topic}`,
      `Challenges in ${topic}`,
      `Future of ${topic}`
    ];
    const cross = this.context.lastTopic && this.context.lastTopic !== topic ?
      [`How does ${topic} relate to ${this.context.lastTopic}?`] : [];
    const entityChips = entities.slice(0, 2).map(e => `Deep dive on ${e.keyword}?`);

    return shuffle(uniq([...semantic, ...refine, ...stock, ...cross, ...entityChips])).slice(0, 5);
  }

  /**
   * Returns a blank response for empty queries.
   * @returns {object} - Blank response object.
   */
  blank() {
    return { main: "Ask me anything!", details: [], stats: null, quote: null, related: [] };
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

/* ---------- Export ---------- */
global.NovaAI = NovaAI;

})(typeof window !== "undefined" ? window : this);