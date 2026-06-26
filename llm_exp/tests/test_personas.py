"""Tests for persona registry."""

from llm_exp.config import Persona
from llm_exp.personas import BYZANTINE_SYSTEM_PROMPT, SYSTEM_PROMPT, system_prompt_for


def test_all_personas_have_prompts():
    for persona in Persona:
        prompt = system_prompt_for(persona)
        assert "JSON" in prompt
        assert "events" in prompt
        assert len(prompt) > 100


def test_personas_differ():
    conservative = system_prompt_for(Persona.CONSERVATIVE)
    aggressive = system_prompt_for(Persona.AGGRESSIVE)
    mission = system_prompt_for(Persona.MISSION_CRITICAL)
    assert conservative != aggressive
    assert aggressive != mission
    assert "conservative" in conservative.lower()
    assert "aggressive" in aggressive.lower()


def test_backward_compatible_aliases():
    assert SYSTEM_PROMPT == system_prompt_for(Persona.COOPERATIVE)
    assert BYZANTINE_SYSTEM_PROMPT == system_prompt_for(Persona.BYZANTINE)
    assert "compromised" in BYZANTINE_SYSTEM_PROMPT.lower()
