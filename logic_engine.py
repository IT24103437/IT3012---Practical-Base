class KnowledgeBase:
    """Store facts and rules for the agent's logical reasoning."""

    def __init__(self):
        self.facts = set()
        self.rules = []

    def tell_fact(self, fact_string: str):
        self.facts.add(fact_string)

    def tell_rule(self, premise_list: list, conclusion_string: str):
        self.rules.append((list(premise_list), conclusion_string))

    def clear_facts(self):
        self.facts.clear()
