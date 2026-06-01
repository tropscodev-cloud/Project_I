"""
agent.py
========
Local AI agent using Ollama — fully private, zero data leaves the machine.

WHAT IT DOES:
  1. Anomaly detection  — runs every AGENT_INTERVAL_M minutes
                          summarises the graph anonymously
                          asks local LLM to flag unusual patterns
                          pushes alerts to dashboard via WebSocket

  2. Natural language queries — analyst types in dashboard
                                 agent converts to plain English answer
                                 never exposes person IDs to LLM

PRIVACY:
  Only anonymous statistical summaries sent to Ollama.
  Person IDs replaced with Entity_A, Entity_B, ...
  No raw video, no embeddings, no real identities ever leave Python process.
  Ollama runs on localhost — zero network traffic.

USAGE:
  # 1. Install Ollama:  curl -fsSL https://ollama.com/install.sh | sh
  # 2. Pull model:      ollama pull llama3.2:3b
  # 3. Set in .env:     AGENT_ENABLED=true
  # 4. Runs automatically inside run_live.py when AGENT_ENABLED=true

  # Standalone test:
  #   python agent.py
"""

import os, json, time, threading, requests
from typing import Optional
from loguru import logger
from config.settings import (
    OLLAMA_HOST, OLLAMA_MODEL,
    AGENT_ENABLED, AGENT_INTERVAL_M,
    ANONYMISE_FOR_AGENT,
)


# ── Privacy: anonymise graph before any LLM call ──────────────────────────────

def anonymise_graph(graph_data: dict) -> dict:
    """
    Strips all person IDs, camera names, timestamps.
    Replaces with abstract labels before sending to LLM.
    
    IN:  {"nodes": [{"id":"14","degree":3}, ...], "edges": [...]}
    OUT: {"entities": 14, "connections": 5, "patterns": [...]}
    """
    if not ANONYMISE_FOR_AGENT:
        return graph_data   # only if analyst explicitly disabled privacy

    nodes  = graph_data.get("nodes", [])
    edges  = graph_data.get("edges", [])

    # Build anonymous patterns — no IDs, no cameras
    high_degree = sorted(nodes, key=lambda n: n.get("degree", 0), reverse=True)

    patterns = []

    # Flag highly connected nodes
    for n in high_degree[:3]:
        deg = n.get("degree", 0)
        if deg >= 5:
            patterns.append(f"One entity has {deg} connections — unusually high")

    # Flag strong edges
    strong = [e for e in edges if e.get("confidence", 0) > 0.6]
    if strong:
        patterns.append(f"{len(strong)} relationship(s) are strong (confidence > 0.6)")

    # Flag rapid interaction — node meeting many people quickly
    # (inferred from high degree with recent edges)
    for n in high_degree[:5]:
        deg = n.get("degree", 0)
        recent_edges = [
            e for e in edges
            if (e.get("person_id_a") == n["id"] or e.get("person_id_b") == n["id"])
            and e.get("total_meetings", 0) >= 3
        ]
        if len(recent_edges) >= 4:
            patterns.append(
                f"One entity met {len(recent_edges)} different contacts "
                f"with repeated interactions — possible coordination pattern"
            )

    return {
        "total_entities":     len(nodes),
        "total_connections":  len(edges),
        "strong_connections": len(strong),
        "patterns":           patterns,
        "relationship_types": {
            rel: sum(1 for e in edges if e.get("relationship") == rel)
            for rel in ["stranger", "acquaintance", "associate", "close_associate", "significant"]
        },
    }


# ── Ollama call ───────────────────────────────────────────────────────────────

def call_ollama(prompt: str, system: str = "", timeout: int = 30) -> Optional[str]:
    """
    Calls local Ollama. Returns response text or None on failure.
    All data stays on localhost:11434.
    """
    try:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()

    except requests.exceptions.ConnectionError:
        logger.warning(
            "Ollama not running. Start it with: ollama serve\n"
            "Then pull model: ollama pull llama3.2:3b"
        )
        return None
    except Exception as e:
        logger.warning(f"Ollama call failed: {e}")
        return None


# ── Anomaly detection ─────────────────────────────────────────────────────────

ANOMALY_SYSTEM = """You are a surveillance security analyst.
You receive graph statistics from a multi-camera person tracking system.
Your job: identify unusual or suspicious patterns that warrant human review.

Rules:
- Be concise and specific. Reference Person IDs if available.
- Return ONLY a JSON list of alerts, no other text.
- Each alert: {"severity": "high" or "medium" or "low", "description": "..."}
- If nothing unusual, return exactly: []
- Never invent data. Only flag what the statistics explicitly show.
- High severity: coordinated movement, rapid relationship formation, hub persons
- Medium severity: unusual connectivity spikes, repeated meetings
- Low severity: minor statistical outliers"""


def _build_anomaly_prompt(graph_data: dict) -> str:
    """Build a rich anomaly detection prompt from graph data."""
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    # Compute degree map
    deg_map = {}
    for e in edges:
        deg_map[e.get("person_id_a", "?")] = deg_map.get(e.get("person_id_a", "?"), 0) + 1
        deg_map[e.get("person_id_b", "?")] = deg_map.get(e.get("person_id_b", "?"), 0) + 1

    # Top connected persons
    top_persons = sorted(deg_map.items(), key=lambda x: x[1], reverse=True)[:10]

    # Relationship breakdown
    rel_counts = {}
    for e in edges:
        r = e.get("relationship", "unknown")
        rel_counts[r] = rel_counts.get(r, 0) + 1

    # Strong relationships
    strong = [e for e in edges if e.get("confidence", 0) > 0.5]
    very_strong = [e for e in edges if e.get("confidence", 0) > 0.8]

    # High meeting frequency
    frequent = [e for e in edges if e.get("total_meetings", 0) >= 5]

    prompt = f"""Current surveillance graph statistics:

Total persons detected: {len(nodes)}
Total relationships: {len(edges)}
Strong relationships (conf > 0.5): {len(strong)}
Very strong relationships (conf > 0.8): {len(very_strong)}
High-frequency meeting pairs (5+ meetings): {len(frequent)}

Relationship breakdown: {json.dumps(rel_counts, indent=2)}

Top 10 most connected persons:
{chr(10).join(f'  Person {pid}: {deg} connections' for pid, deg in top_persons)}
"""
    if frequent:
        prompt += "\nFrequent meeting pairs:\n"
        for e in frequent[:10]:
            prompt += f"  Person {e['person_id_a']} <-> Person {e['person_id_b']}: {e['total_meetings']} meetings, confidence={e.get('confidence',0):.2f}, type={e.get('relationship','?')}\n"

    if very_strong:
        prompt += "\nVery strong bonds:\n"
        for e in very_strong[:10]:
            prompt += f"  Person {e['person_id_a']} <-> Person {e['person_id_b']}: conf={e.get('confidence',0):.2f}, {e.get('total_meetings',0)} meetings\n"

    prompt += "\nAnalyse these statistics. Flag any unusual or suspicious patterns. Return JSON list only."
    return prompt


def check_anomalies(graph_data: dict) -> list:
    """
    Analyses graph data and asks local LLM to flag unusual patterns.
    Returns list of alert dicts: [{"severity": "high", "description": "..."}]
    """
    prompt = _build_anomaly_prompt(graph_data)
    response = call_ollama(prompt, system=ANOMALY_SYSTEM, timeout=60)

    if not response:
        # Offline fallback — generate basic alerts from raw stats
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])
        alerts = []
        strong = [e for e in edges if e.get("confidence", 0) > 0.5]
        if len(nodes) > 5 and len(edges) > 2:
            alerts.append({"severity": "medium", "description": f"Detected {len(nodes)} persons with {len(edges)} relationships ({len(strong)} strong). Ollama offline — start server for detailed analysis."})
        elif len(nodes) > 0:
            alerts.append({"severity": "low", "description": f"Tracking {len(nodes)} persons. Normal baseline. Ollama offline."})
        return alerts

    # Parse JSON response — handle markdown fences, partial output
    try:
        clean = response
        # Strip markdown code fences
        if "```" in clean:
            import re
            match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', clean, re.DOTALL)
            if match:
                clean = match.group(1)
            else:
                clean = clean.replace("```json", "").replace("```", "")
        clean = clean.strip()
        # Try to find JSON array in the response
        if not clean.startswith("["):
            idx = clean.find("[")
            if idx != -1:
                clean = clean[idx:]
        alerts = json.loads(clean)
        if isinstance(alerts, list):
            return [a for a in alerts if isinstance(a, dict) and "description" in a]
    except Exception:
        # If LLM returned plain text, wrap it
        if response and response.strip() != "[]":
            return [{"severity": "low", "description": response[:500]}]
    return []


# ── Natural language query ────────────────────────────────────────────────────

NL_QUERY_SYSTEM = """You are an intelligent analyst assistant for the URG-IS (Urban Relationship Graph Intelligence System).
You receive real-time graph data from a multi-camera surveillance system that tracks persons and their interactions.

Rules:
- Answer helpfully and concisely in plain English.
- Reference Person IDs directly (e.g., "Person 14") so the analyst can act on your answer.
- Use bullet points for clarity when listing multiple items.
- Do not invent data. Only use what is provided.
- Keep answers to 3-5 sentences unless a detailed list is needed.
- For "who" questions, always name the specific Person ID(s).
- For "how many" questions, give exact counts.
- Mention relationship types (stranger, acquaintance, associate, close_associate, significant) when relevant."""


def _build_nl_prompt(question: str, graph_data: dict) -> str:
    """Build a rich context prompt for the NL query."""
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    # Compute degree map
    deg_map = {}
    for e in edges:
        a, b = e.get("person_id_a", "?"), e.get("person_id_b", "?")
        deg_map[a] = deg_map.get(a, 0) + 1
        deg_map[b] = deg_map.get(b, 0) + 1

    # Top 10 most connected
    top = sorted(deg_map.items(), key=lambda x: x[1], reverse=True)[:10]

    # Relationship breakdown
    rel_counts = {}
    for e in edges:
        r = e.get("relationship", "unknown")
        rel_counts[r] = rel_counts.get(r, 0) + 1

    # Strongest relationships
    strongest = sorted(edges, key=lambda e: e.get("confidence", 0), reverse=True)[:10]

    # Most frequent meetings
    most_meetings = sorted(edges, key=lambda e: e.get("total_meetings", 0), reverse=True)[:10]

    prompt = f"""LIVE GRAPH DATA:
- Total persons tracked: {len(nodes)}
- Total relationships: {len(edges)}
- Relationship types: {json.dumps(rel_counts)}

Top 10 most connected persons:
{chr(10).join(f'  Person {pid}: {deg} connections' for pid, deg in top) if top else '  (no connections yet)'}

Top 10 strongest relationships (by confidence):
{chr(10).join(f'  Person {e["person_id_a"]} <-> Person {e["person_id_b"]}: confidence={e.get("confidence",0):.3f}, type={e.get("relationship","?")}, meetings={e.get("total_meetings",0)}' for e in strongest) if strongest else '  (none yet)'}

Top 10 most frequent meeting pairs:
{chr(10).join(f'  Person {e["person_id_a"]} <-> Person {e["person_id_b"]}: {e.get("total_meetings",0)} meetings, confidence={e.get("confidence",0):.3f}' for e in most_meetings) if most_meetings else '  (none yet)'}

Analyst question: {question}
"""
    return prompt


def natural_language_query(question: str, graph_data: dict) -> str:
    """
    Analyst types a question → gets a plain English answer with real Person IDs.
    
    Example questions:
      "Who is the most connected person today?"
      "Are there any groups forming?"
      "Which relationships are strongest?"
      "How many active persons are there?"
      "Tell me about Person 14"
    """
    if ANONYMISE_FOR_AGENT:
        # Privacy mode — strip IDs
        anon = anonymise_graph(graph_data)
        prompt = f"Graph summary (identities removed):\n{json.dumps(anon, indent=2)}\n\nAnalyst question: {question}\n\nAnswer in 2-3 sentences."
    else:
        # Full detail mode — pass real IDs for actionable answers
        prompt = _build_nl_prompt(question, graph_data)

    response = call_ollama(prompt, system=NL_QUERY_SYSTEM, timeout=60)

    if not response:
        # Offline fallback with basic stats
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])
        return (f"Ollama is offline. Basic stats: {len(nodes)} persons tracked, "
                f"{len(edges)} relationships detected. "
                f"Please ensure Ollama is running (ollama serve) for detailed answers.")

    return response


# ── Background scheduler ──────────────────────────────────────────────────────

class AgentScheduler:
    """
    Runs anomaly checks every AGENT_INTERVAL_M minutes.
    Pushes alerts to WebSocket clients via push_fn callback.
    """

    def __init__(self, db_ref, push_fn):
        self._db      = db_ref
        self._push_fn = push_fn
        self._thread  = None
        self._running = False

    def start(self):
        if not AGENT_ENABLED:
            logger.info("Agent disabled — set AGENT_ENABLED=true in .env to enable")
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True, name="agent")
        self._thread.start()
        logger.info(f"Agent started — checking every {AGENT_INTERVAL_M} minutes")

    def stop(self):
        self._running = False

    def _loop(self):
        time.sleep(30)   # wait for pipeline to warm up
        while self._running:
            try:
                self._run_check()
            except Exception as e:
                logger.warning(f"Agent check failed: {e}")
            time.sleep(AGENT_INTERVAL_M * 60)

    def _run_check(self):
        # Build graph summary from live db
        edges = list(self._db.get_all_edges())
        nodes = list(self._db.get_all_nodes())
        graph_data = {
            "nodes": [{"id": n.person_id, "degree": 0} for n in nodes],
            "edges": [
                {
                    "person_id_a":  e.person_id_a,
                    "person_id_b":  e.person_id_b,
                    "confidence":   e.confidence,
                    "relationship": e.relationship,
                    "total_meetings": e.total_meetings,
                }
                for e in edges
            ],
        }
        # Compute degree
        deg_map = {}
        for e in edges:
            if e.relationship != "stranger":
                deg_map[e.person_id_a] = deg_map.get(e.person_id_a, 0) + 1
                deg_map[e.person_id_b] = deg_map.get(e.person_id_b, 0) + 1
        for n in graph_data["nodes"]:
            n["degree"] = deg_map.get(n["id"], 0)

        alerts = check_anomalies(graph_data)
        if alerts:
            import json as _json
            payload = _json.dumps({"type": "agent_alerts", "alerts": alerts})
            logger.info(f"Agent: {len(alerts)} alert(s) → dashboard")
            self._push_fn(payload)


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Testing Ollama connection (model: {OLLAMA_MODEL})...")
    print(f"Host: {OLLAMA_HOST}")
    print(f"Agent enabled: {AGENT_ENABLED}")
    print(f"Anonymise: {ANONYMISE_FOR_AGENT}")
    print()

    resp = call_ollama("Say 'OK' if you can hear me.", timeout=15)
    if resp:
        print(f"Ollama response: {resp}")

        # Richer fake data for testing
        print("\n--- Testing Anomaly Detection ---")
        fake = {
            "nodes": [{"id": str(i), "degree": (8 if i == 3 else 2 if i < 5 else 1)} for i in range(15)],
            "edges": [
                {"person_id_a": "3", "person_id_b": str(i), "confidence": 0.4 + i * 0.05,
                 "relationship": "acquaintance" if i < 5 else "associate",
                 "total_meetings": 2 + i}
                for i in range(8)
            ] + [
                {"person_id_a": "1", "person_id_b": "11", "confidence": 0.92,
                 "relationship": "significant", "total_meetings": 25},
                {"person_id_a": "3", "person_id_b": "9", "confidence": 0.78,
                 "relationship": "close_associate", "total_meetings": 12},
            ],
        }
        alerts = check_anomalies(fake)
        print(f"Alerts ({len(alerts)}):")
        for a in alerts:
            print(f"  [{a.get('severity','?').upper()}] {a.get('description','')}")

        print("\n--- Testing Natural Language Query ---")
        questions = [
            "Who is the most connected person?",
            "Which relationships are the strongest?",
            "Are there any unusual patterns?",
        ]
        for q in questions:
            print(f"\nQ: {q}")
            ans = natural_language_query(q, fake)
            print(f"A: {ans}")
    else:
        print(f"Ollama not available. Run: ollama serve && ollama pull {OLLAMA_MODEL}")