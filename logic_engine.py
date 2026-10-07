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

    def forward_chain(self):
        """Infer conclusions until no rule adds a new fact."""
        new_facts_added = True

        while new_facts_added:
            new_facts_added = False

            for premises, conclusion in self.rules:
                if conclusion not in self.facts:
                    if all(premise in self.facts for premise in premises):
                        self.facts.add(conclusion)
                        new_facts_added = True
