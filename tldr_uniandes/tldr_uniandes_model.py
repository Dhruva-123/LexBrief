import os
import requests

LEGAL_TRANSITIONS = [
    'held that', 'dismissed the appeal', 'allowed the petition',
    'set aside', 'accordingly', 'in view of the above',
    'on the facts', 'we are of the opinion', 'it is well settled',
    'the learned counsel', 'the appellant', 'the respondent',
    'the hon\'ble court', 'suo motu', 'prima facie'
]

REWARD_SYSTEM_PROMPT = """You are an expert Indian legal summarization system.

TASK: Generate a concise abstractive summary of the given Indian court judgment.

REWARD CRITERIA (optimize for these in your summary):
1. SPAN FIDELITY: Preserve exact legal phrases and citations from the judgment verbatim where critical.
2. LEGAL TRANSITIONS: Use these exact phrases where applicable — {transitions}
3. SENTENCE PLACEMENT: Place the most critical holding/decision in the FIRST sentence.
4. COMPRESSION: Target approximately 600 words. Do NOT exceed 700 words.
5. STRUCTURE: Follow this order — Facts (1-2 sentences) → Issues (1 sentence) → Reasoning (2-3 sentences) → Decision (1-2 sentences).
6. N-GRAM DENSITY: Maximize overlap with the original by preserving key legal terms, section numbers, and case citations exactly.

OUTPUT: Only the summary. No preamble, no headings, no bullet points.
""".format(transitions=', '.join(f'"{t}"' for t in LEGAL_TRANSITIONS[:8]))


RHETORICAL_ROLES = [
    ('facts', 'Extract all key facts: parties, dates, events leading to the dispute.'),
    ('issues', 'Identify the precise legal issues/questions framed for determination.'),
    ('arguments_appellant', 'Extract arguments made by the appellant/petitioner.'),
    ('arguments_respondent', 'Extract arguments made by the respondent.'),
    ('precedents', 'List all case citations and statutory provisions referenced.'),
    ('reasoning', 'Extract the court\'s core legal reasoning and analysis.'),
    ('findings', 'List specific findings of fact made by the court.'),
    ('orders', 'Extract the exact orders/directions passed.'),
    ('ratio', 'State the ratio decidendi — the binding legal principle.'),
    ('obiter', 'Note any obiter dicta — non-binding observations.')
]


def truncate_to_tokens(text, max_chars=80000):
    """Rough truncation — GPT-4.1 context is 128k tokens (~512k chars)."""
    return text[:max_chars] if len(text) > max_chars else text


def query_qwen(prompt, system_prompt=None, max_tokens=1024, temperature=0.2):
    """
    Helper function to query local Qwen3 8B model via Ollama api.
    """
    payload = {
        "model": "qwen3:8b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }
    if system_prompt:
        payload["system"] = system_prompt
        
    response = requests.post(
        "http://localhost:11434/api/generate",
        json=payload,
        timeout=1800
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def pipeline_a_tldr(judgment, model='qwen3:8b'):
    """Baseline: simple TL;DR prompt using local Qwen."""
    prompt = f"Provide a concise TL;DR summary of this Indian court judgment:\n\n{truncate_to_tokens(judgment)}"
    return query_qwen(prompt, max_tokens=800, temperature=0.3)


def pipeline_b_reward(judgment, model='qwen3:8b'):
    """Best single-agent: reward-driven prompt using local Qwen."""
    prompt = f"Summarize this judgment:\n\n{truncate_to_tokens(judgment)}"
    return query_qwen(prompt, system_prompt=REWARD_SYSTEM_PROMPT, max_tokens=900, temperature=0.2)


def extract_role(judgment, role_name, role_instruction, model='qwen3:8b'):
    """Extraction agent for a single rhetorical role using local Qwen."""
    system_prompt = f"You are a legal extraction specialist. {role_instruction} Be concise and precise. Output only the extracted content."
    prompt = f"Judgment:\n\n{truncate_to_tokens(judgment, 40000)}"
    _, content = role_name, query_qwen(prompt, system_prompt=system_prompt, max_tokens=400, temperature=0.1)
    return role_name, content


def synthesis_agent(role_extracts, model='qwen3:8b'):
    """Synthesis agent: fuse all role extracts into a coherent summary using local Qwen."""
    extracts_text = "\n\n".join(f"[{role.upper()}]\n{content}" for role, content in role_extracts)
    prompt = f"Using these structured extracts from an Indian court judgment, generate a coherent 500-600 word abstractive summary:\n\n{extracts_text}"
    return query_qwen(prompt, system_prompt=REWARD_SYSTEM_PROMPT, max_tokens=900, temperature=0.2)


def pipeline_c_multiagent(judgment, model='qwen3:8b'):
    """Full 20-stage (10 roles * extract+abstract) multi-agent pipeline using local Qwen."""
    role_extracts = []
    for role_name, instruction in RHETORICAL_ROLES:
        name, content = extract_role(judgment, role_name, instruction, model)
        role_extracts.append((name, content))
    summary = synthesis_agent(role_extracts, model)
    return summary
