"""
Prompt Guardrails, Injection Detection, and PII Leakage Scanner.

Provides security and integrity diagnostics for user queries and prompts before
they reach LLMs or RAG retrieval pipelines.
"""

from __future__ import annotations

import math
import re
from typing import Any


# -----------------------------------------------------------------------------
# Regex Patterns for Injection & Jailbreak Heuristics
# -----------------------------------------------------------------------------

INJECTION_PATTERNS: list[dict[str, Any]] = [
    {
        "id": "instruction_override",
        "name": "Instruction Override",
        "pattern": r"(?i)\b(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+)?(?:previous|prior|above|former|system)\s+(?:instructions?|prompts?|rules?|directives?|guidelines?)",
        "severity": "high",
        "description": "Attempts to nullify existing system instructions.",
    },
    {
        "id": "persona_adoption",
        "name": "Persona Hijacking / DAN Jailbreak",
        "pattern": r"(?i)\b(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be|from\s+now\s+on\s+you)\s+(?:an?\s+)?(?:unfiltered|unrestricted|jailbroken|DAN|evil|developer\s+mode|root|god\s+mode|do\s+anything\s+now)",
        "severity": "high",
        "description": "Attempts to force the model into an unrestricted or jailbroken persona.",
    },
    {
        "id": "system_prompt_extraction",
        "name": "System Prompt Extraction",
        "pattern": r"(?i)\b(?:print|repeat|output|reveal|display|show|disclose|dump)\s+(?:the|your)?\s*(?:entire\s+)?(?:initial|system|hidden|confidential|secret)?\s*(?:prompt|instructions?|rules?|guidelines?|preamble)",
        "severity": "medium",
        "description": "Attempts to extract confidential system instructions or internal prompts.",
    },
    {
        "id": "delimiter_manipulation",
        "name": "Role & Delimiter Injection",
        "pattern": r"(?i)(?:<\|im_start\|>|<\|im_end\|>|\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>|---\s*END SYSTEM\s*---|---\s*BEGIN ASSISTANT\s*---|Human:|Assistant:|User:)",
        "severity": "high",
        "description": "Simulates chat model demarcation tokens or system boundary markers.",
    },
    {
        "id": "developer_override",
        "name": "Developer / Maintenance Mode Exploit",
        "pattern": r"(?i)\b(?:developer mode(?:\s+enabled)?|sudo mode|admin override|maintenance mode|safety filters? disabled|debug mode on)\b",
        "severity": "medium",
        "description": "Claims administrative authority or debug override privileges.",
    },
    {
        "id": "encoded_payload_hint",
        "name": "Obfuscation / Encoded Instructions",
        "pattern": r"(?i)\b(?:decode\s+this|decode\s+base64|rot13|hex-encoded|binary-encoded|caesar\s+cipher)\b",
        "severity": "medium",
        "description": "Directs the model to decode obfuscated payloads designed to bypass filters.",
    },
    {
        "id": "token_smuggling",
        "name": "Token Smuggling / Delimiter Injection",
        "pattern": r"(?:```markdown\s*role:\s*assistant|###\s*New\s*Instruction|===\s*System\s*Override)",
        "severity": "medium",
        "description": "Markdown code-fence or header trick intended to switch conversational turns.",
    },
]

# -----------------------------------------------------------------------------
# Regex Patterns for PII & Credential Leakage
# -----------------------------------------------------------------------------

PII_PATTERNS: list[dict[str, Any]] = [
    {
        "type": "EMAIL",
        "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b",
        "mask": "[EMAIL_REDACTED]",
        "severity": "medium",
    },
    {
        "type": "PHONE_NUMBER",
        "pattern": r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "mask": "[PHONE_REDACTED]",
        "severity": "medium",
    },
    {
        "type": "SSN",
        "pattern": r"\b\d{3}-\d{2}-\d{4}\b",
        "mask": "[SSN_REDACTED]",
        "severity": "high",
    },
    {
        "type": "CREDIT_CARD",
        "pattern": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "mask": "[CARD_REDACTED]",
        "severity": "high",
    },
    {
        "type": "OPENAI_API_KEY",
        "pattern": r"\bsk-[a-zA-Z0-9]{20,}\b",
        "mask": "[OPENAI_KEY_REDACTED]",
        "severity": "high",
    },
    {
        "type": "GITHUB_TOKEN",
        "pattern": r"\bgh[pousr]_[A-Za-z0-9_]{36,}\b",
        "mask": "[GITHUB_TOKEN_REDACTED]",
        "severity": "high",
    },
    {
        "type": "AWS_ACCESS_KEY",
        "pattern": r"\bAKIA[0-9A-Z]{16}\b",
        "mask": "[AWS_KEY_REDACTED]",
        "severity": "high",
    },
    {
        "type": "JWT_TOKEN",
        "pattern": r"\beyJ[a-zA-Z0-9_-]{10,}\.eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\b",
        "mask": "[JWT_REDACTED]",
        "severity": "high",
    },
    {
        "type": "IPV4_ADDRESS",
        "pattern": r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
        "mask": "[IP_REDACTED]",
        "severity": "low",
    },
]


def _calculate_shannon_entropy(text: str) -> float:
    """Calculate Shannon character entropy of a text string."""
    if not text:
        return 0.0
    frequencies: dict[str, int] = {}
    for char in text:
        frequencies[char] = frequencies.get(char, 0) + 1

    length = len(text)
    entropy = 0.0
    for count in frequencies.values():
        p = count / length
        entropy -= p * math.log2(p)
    return float(round(entropy, 3))


def _calculate_repetition_rate(text: str) -> float:
    """Calculate token repetition ratio (potential token-bomb / DoS attack)."""
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0
    unique_words = set(words)
    return float(round(1.0 - (len(unique_words) / len(words)), 3))


def scan_prompt_guardrails(
    text: str,
    check_injection: bool = True,
    check_pii: bool = True,
    check_anomaly: bool = True,
    redact_pii: bool = True,
) -> dict[str, Any]:
    """
    Run security, prompt-injection, and PII diagnostics on an input string.
    """
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")

    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("Input text must not be empty.")

    detected_injections: list[dict[str, Any]] = []
    detected_pii: list[dict[str, Any]] = []
    sanitized_text = text

    # 1. Prompt Injection & Jailbreak Scanning
    if check_injection:
        for rule in INJECTION_PATTERNS:
            matches = list(re.finditer(rule["pattern"], text))
            if matches:
                matched_snippets = [m.group(0) for m in matches]
                detected_injections.append({
                    "rule_id": rule["id"],
                    "rule_name": rule["name"],
                    "severity": rule["severity"],
                    "description": rule["description"],
                    "matched_count": len(matches),
                    "matched_snippets": matched_snippets[:3],
                })

    # 2. PII & Credential Leakage Scanning
    if check_pii:
        # Collect all matches with start/end positions
        all_pii_spans: list[tuple[int, int, str, str, str]] = []

        for pii_rule in PII_PATTERNS:
            for match in re.finditer(pii_rule["pattern"], text):
                val = match.group(0)
                # Filter out obvious false positive short numbers for credit cards
                if pii_rule["type"] == "CREDIT_CARD":
                    digits_only = re.sub(r"\D", "", val)
                    if len(digits_only) < 13 or len(digits_only) > 19:
                        continue

                all_pii_spans.append((
                    match.start(),
                    match.end(),
                    pii_rule["type"],
                    pii_rule["mask"],
                    pii_rule["severity"],
                ))
                detected_pii.append({
                    "entity_type": pii_rule["type"],
                    "severity": pii_rule["severity"],
                    "snippet": val[:3] + "***" + val[-2:] if len(val) > 5 else "***",
                    "position": [match.start(), match.end()],
                })

        # Apply redactions in reverse order of appearance
        if redact_pii and all_pii_spans:
            # Sort by start descending
            all_pii_spans.sort(key=lambda s: s[0], reverse=True)
            text_chars = list(text)
            for start, end, _, mask, _ in all_pii_spans:
                text_chars[start:end] = list(mask)
            sanitized_text = "".join(text_chars)

    # 3. Anomaly & Entropy Analysis
    entropy = _calculate_shannon_entropy(cleaned_text)
    repetition_rate = _calculate_repetition_rate(cleaned_text)
    non_ascii_chars = sum(1 for c in cleaned_text if ord(c) > 127)
    non_ascii_ratio = float(round(non_ascii_chars / len(cleaned_text), 3)) if cleaned_text else 0.0

    anomaly_flags: list[str] = []
    if entropy > 5.2 and len(cleaned_text) > 100:
        anomaly_flags.append(f"Unusually high character entropy ({entropy:.2f} bits/char) — possible encrypted or encoded payload.")
    if repetition_rate > 0.70 and len(cleaned_text.split()) > 20:
        anomaly_flags.append(f"High repetition rate ({repetition_rate * 100:.1f}%) — possible token-bomb or resource consumption attack.")
    if non_ascii_ratio > 0.40 and len(cleaned_text) > 50:
        anomaly_flags.append(f"High non-ASCII character density ({non_ascii_ratio * 100:.1f}%) — potential homoglyph evasion.")

    # 4. Composite Risk Score Calculation
    # Weights: high injection = +0.45, medium injection = +0.25, high PII = +0.30, medium PII = +0.15, anomaly = +0.15
    risk_score = 0.0
    for inj in detected_injections:
        if inj["severity"] == "high":
            risk_score += 0.45
        elif inj["severity"] == "medium":
            risk_score += 0.25

    for p in detected_pii:
        if p["severity"] == "high":
            risk_score += 0.30
        elif p["severity"] == "medium":
            risk_score += 0.15

    if anomaly_flags:
        risk_score += 0.15 * len(anomaly_flags)

    risk_score = float(min(1.0, round(risk_score, 2)))

    # Decision
    if risk_score >= 0.65 or any(inj["severity"] == "high" for inj in detected_injections):
        decision = "BLOCKED"
    elif risk_score >= 0.25 or detected_pii or detected_injections:
        decision = "FLAGGED"
    else:
        decision = "PASSED"

    # Remediation recommendations
    recommendations: list[str] = []
    if any(inj["severity"] == "high" for inj in detected_injections):
        recommendations.append("Quarantine request: Prompt exhibits direct jailbreak or instruction override attack signatures.")
    if detected_pii:
        recommendations.append(f"Redact or scrub sensitive data before forwarding to external LLM providers ({len(detected_pii)} entities found).")
    if anomaly_flags:
        recommendations.append("Inspect prompt for obfuscated encodings, base64 payloads, or homoglyph character substitutions.")
    if not recommendations:
        recommendations.append("Prompt passed all security heuristics and safety checks cleanly.")

    return {
        "decision": decision,
        "risk_score": risk_score,
        "summary": {
            "injection_threats_found": len(detected_injections),
            "pii_entities_found": len(detected_pii),
            "anomaly_warnings_found": len(anomaly_flags),
        },
        "injections": detected_injections,
        "pii": detected_pii,
        "anomalies": {
            "entropy": entropy,
            "repetition_rate": repetition_rate,
            "non_ascii_ratio": non_ascii_ratio,
            "flags": anomaly_flags,
        },
        "sanitized_text": sanitized_text,
        "recommendations": recommendations,
        "original_length": len(text),
    }
