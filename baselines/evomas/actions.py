"""Dependency-free helpers shared by the EvoMAS runner and offline scorer."""
from __future__ import annotations

import re

DOMAINS = [
    "customer_relationship_manager", "project_management", "calendar",
    "email", "analytics", "company_directory",
]


def extract_calls(output: str) -> list[str]:
    if not output:
        return []
    section = output.split("FUNCTION_CALLS:")[-1] if "FUNCTION_CALLS:" in output else output
    formats = (
        r"(\w+\.\w+\.func\([^)]*\))",
        r"(\w+\.\w+\([^)]*\))",
        r"(\w+_\w+(?:_\w+)*\([^)]*\))",
    )
    matches: list[str] = []
    for line in section.splitlines():
        line = line.strip()
        if not line or line.startswith(("```", "#")):
            continue
        for pattern in formats:
            found = re.findall(pattern, line)
            if found:
                matches.extend(found)
                break
    if not matches:
        for pattern in formats:
            matches = re.findall(pattern, section, re.DOTALL)
            if matches:
                break
    out: list[str] = []
    for call in matches:
        if ".func(" in call:
            out.append(call)
            continue
        if "." in call.split("(", 1)[0]:
            out.append(call.replace("(", ".func(", 1))
            continue
        for domain in DOMAINS:
            prefix = domain + "_"
            if call.startswith(prefix):
                out.append(f"{domain}.{call[len(prefix):].replace('(', '.func(', 1)}")
                break
        else:
            out.append(call)
    return out
