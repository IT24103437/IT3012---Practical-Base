from agent import SearchAgent
from logic_engine import KnowledgeBase


def test_forward_chaining():
    kb = KnowledgeBase()

    # Add Domain Rules
    kb.tell_rule(['TargetVisible', 'HasDust'], 'SafeToEngage')
    kb.tell_rule(['SafeToEngage', 'BloodseekerMissing'], 'Retreat')

    # Test Case 1: Safe Engagement
    kb.clear_facts()
    kb.tell_fact('TargetVisible')
    kb.tell_fact('HasDust')
    kb.forward_chain()
    assert 'SafeToEngage' in kb.facts, "Test 1 Failed: Should deduce SafeToEngage"
    assert 'Retreat' not in kb.facts, "Test 1 Failed: Should NOT deduce Retreat"

    # Test Case 2: Unsafe Engagement (Bloodseeker Missing)
    kb.clear_facts()
    kb.tell_fact('TargetVisible')
    kb.tell_fact('HasDust')
    kb.tell_fact('BloodseekerMissing')
    kb.forward_chain()
    assert 'Retreat' in kb.facts, "Test 2 Failed: Should deduce Retreat to override search"

    print("All Logic Engine Test Cases Passed!")


def test_rules_requiring_another_pass():
    kb = KnowledgeBase()
    kb.tell_rule(['SafeToEngage', 'BloodseekerMissing'], 'Retreat')
    kb.tell_rule(['TargetVisible', 'HasDust'], 'SafeToEngage')
    for fact in ['TargetVisible', 'HasDust', 'BloodseekerMissing']:
        kb.tell_fact(fact)
    kb.forward_chain()
    assert 'Retreat' in kb.facts


def test_astar_avoids_unsafe_tile():
    agent = SearchAgent()
    tile_facts = {
        (1, 0): ['TargetVisible', 'HasDust', 'BloodseekerMissing'],
        (0, 1): ['TargetVisible', 'HasDust'],
    }

    path = agent.astar_search((0, 0), (2, 0), [], (3, 2), tile_facts=tile_facts)
    assert path == ['Up', 'Right', 'Right', 'Down']
    assert agent.astar_search((0, 0), (0, 1), [], (2, 2), tile_facts=tile_facts) == ['Up']
    assert agent.astar_search((0, 0), (1, 0), [], (2, 1), tile_facts=tile_facts) == []


def test_agent_uses_sample_tile_facts():
    agent = SearchAgent()
    percept = {'all_food': [(2, 0)], 'walls': [], 'grid_size': (3, 2)}
    visited = []

    for _ in range(4):
        agent.sense_and_act(percept)
        visited.append(agent.current_pos)

    assert visited == [(0, 1), (1, 1), (2, 1), (2, 0)]
    assert (1, 0) not in visited


if __name__ == '__main__':
    test_forward_chaining()
    test_rules_requiring_another_pass()
    test_astar_avoids_unsafe_tile()
    test_agent_uses_sample_tile_facts()
    print("A* feasibility and agent integration test cases passed!")
