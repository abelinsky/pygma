from unittest import TestCase
from pygma.agents.pg_agent import BaseAgent, PGAgent


class TestPGAgent(TestCase):
    def test_is_derived_from_BaseAgent(self):
        print("Bases are", PGAgent.__bases__)
        self.assertTrue(issubclass(PGAgent, BaseAgent),
                        'PGAgent is not from BaseAgent class')
