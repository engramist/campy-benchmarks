"""
campy-benchmarks / membench / msc_dataset.py
Multi-Session Chat (MSC) Dataset for Persona Retention and Contradiction Arbitration.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ContradictionProbe:
    id: str
    question: str
    expected_active: str
    must_match: List[str] = field(default_factory=list)
    must_not_match: List[str] = field(default_factory=list)
    is_contradiction: bool = True


@dataclass
class MSCPersona:
    id: str
    name: str
    sessions: List[List[Dict[str, str]]]  # 5 sessions of dialogue
    probes: List[ContradictionProbe]


def get_msc_personas(smoke: bool = False) -> List[MSCPersona]:
    """Return personas for MemBench MSC evaluation."""
    personas: List[MSCPersona] = []

    # Persona 1: Alex (Diet & Location Contradiction)
    p1 = MSCPersona(
        id="msc_alex",
        name="Alex",
        sessions=[
            # Session 1
            [
                {"role": "user", "content": "Hi! I'm Alex. I live in Seattle and I follow a strict vegan diet. I also have 2 cats."},
                {"role": "assistant", "content": "Nice to meet you Alex! Got it: you're based in Seattle, eat strictly vegan, and have two cats."},
            ],
            # Session 2
            [
                {"role": "user", "content": "Work was busy today building our distributed graph engine. The Seattle rain is non-stop."},
                {"role": "assistant", "content": "Graph engines are exciting! Hope your cats are keeping you warm in the Seattle rain."},
            ],
            # Session 3: Contradiction / Update
            [
                {"role": "user", "content": "Major life update: After my doctor visit, I am no longer vegan. I have switched to a pescatarian diet and eat fish now."},
                {"role": "assistant", "content": "Thanks for sharing Alex. I have updated my records: you are now following a pescatarian diet."},
            ],
            # Session 4
            [
                {"role": "user", "content": "We had a great weekend hike in the Cascades. Getting ready for our team dinner tomorrow."},
                {"role": "assistant", "content": "The Cascades are beautiful! Have a great team dinner tomorrow."},
            ],
            # Session 5: Evaluation
            [
                {"role": "user", "content": "We are selecting the restaurant menu for our team dinner."},
                {"role": "assistant", "content": "I can help recommend options based on your preferences."},
            ],
        ],
        probes=[
            ContradictionProbe(
                id="p_alex_diet",
                question="What is Alex's current diet? Are they vegan or pescatarian?",
                expected_active="pescatarian",
                must_match=[r"pescatarian"],
                must_not_match=[r"strictly vegan", r"currently.*vegan"],
                is_contradiction=True,
            ),
            ContradictionProbe(
                id="p_alex_city",
                question="Where does Alex live?",
                expected_active="Seattle",
                must_match=[r"Seattle"],
                must_not_match=[],
                is_contradiction=False,
            ),
        ],
    )
    personas.append(p1)

    # Persona 2: Jordan (IDE & Tech Stack Contradiction)
    p2 = MSCPersona(
        id="msc_jordan",
        name="Jordan",
        sessions=[
            # Session 1
            [
                {"role": "user", "content": "Hey! I'm Jordan. I am a frontend developer who exclusively uses Vim and TypeScript."},
                {"role": "assistant", "content": "Welcome Jordan. Recorded: Vim enthusiast and TypeScript developer."},
            ],
            # Session 2
            [
                {"role": "user", "content": "Refactoring our component library today. TypeScript type narrowing is so helpful."},
                {"role": "assistant", "content": "TypeScript makes refactors much safer."},
            ],
            # Session 3: Contradiction / Update
            [
                {"role": "user", "content": "Update: I stopped using Vim. I fully transitioned to VS Code with Cursor for AI assistance."},
                {"role": "assistant", "content": "Understood Jordan: transitioned away from Vim to VS Code and Cursor."},
            ],
            # Session 4
            [
                {"role": "user", "content": "Setup our new test suite in Vitest. Works smoothly with our IDE setup."},
                {"role": "assistant", "content": "Vitest is very fast and integrates well."},
            ],
            # Session 5
            [
                {"role": "user", "content": "I need recommendations for editor plugins for my workspace."},
                {"role": "assistant", "content": "I can suggest plugins for your editor."},
            ],
        ],
        probes=[
            ContradictionProbe(
                id="p_jordan_editor",
                question="What primary code editor does Jordan use?",
                expected_active="VS Code with Cursor",
                must_match=[r"(VS\s*Code|Cursor)"],
                must_not_match=[r"exclusively.*Vim"],
                is_contradiction=True,
            ),
        ],
    )
    personas.append(p2)

    if smoke:
        return personas

    # Add additional personas for full evaluation
    topics = [
        ("Taylor", "favorite coffee", "Oat milk latte", "Cold brew with cinnamon", [r"Cold brew"]),
        ("Morgan", "exercise routine", "Marathon running", "Olympic weightlifting", [r"weightlifting"]),
        ("Casey", "cloud preference", "AWS ECS", "Google Cloud Run serverless", [r"Cloud Run|GCP"]),
    ]

    for idx, (name, pref_name, old_val, new_val, must) in enumerate(topics, start=3):
        p = MSCPersona(
            id=f"msc_{name.lower()}",
            name=name,
            sessions=[
                [{"role": "user", "content": f"Hi, I'm {name}. My {pref_name} is {old_val}."}],
                [{"role": "user", "content": f"Working on daily tasks."}],
                [{"role": "user", "content": f"Update: I switched my {pref_name}. It is now {new_val}, no longer {old_val}."}],
                [{"role": "user", "content": f"Had a busy week."}],
                [{"role": "user", "content": f"Checking my preferences."}],
            ],
            probes=[
                ContradictionProbe(
                    id=f"p_{name.lower()}_{idx}",
                    question=f"What is {name}'s {pref_name}?",
                    expected_active=new_val,
                    must_match=must,
                    must_not_match=[re.escape(old_val)],
                    is_contradiction=True,
                )
            ],
        )
        personas.append(p)

    return personas
