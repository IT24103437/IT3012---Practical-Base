# agent.py
class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        # If standing directly on food, or just wander / move towards coordinates
        pos = percept['agent_pos']
        # Simple heuristic or fallback random sweep
        return random.choice(self.actions_pool)


class SimpleReflexAgent:
    """React to the current percept using condition-action rules only."""

    def sense_and_act(self, percept: dict) -> str:
        if percept['food_here']:
            return 'Suck'
        elif percept['wall_ahead']:
            return 'Left'
        else:
            return 'Up'


class ModelBasedAgent:
    """Use memory to avoid repeating actions that lead to blocked states."""

    ACTIONS = ('Left', 'Right', 'Up', 'Down')
    DELTAS = {
        'Up': (0, 1),
        'Down': (0, -1),
        'Left': (-1, 0),
        'Right': (1, 0)
    }

    def __init__(self):
        self.position = (0, 0)
        self.facing = 'Up'
        self.visited_cells = {self.position}
        self.blocked_cells = set()
        self.tried_actions = set()
        self.percept_history = []
        self.action_history = []
        self.last_percept = None
        self.last_action = None

    def _next_cell(self, action: str) -> tuple:
        dx, dy = self.DELTAS[action]
        return self.position[0] + dx, self.position[1] + dy

    def _update_state(self, percept: dict) -> bool:
        repeated_percept = percept == self.last_percept

        if self.last_action in self.DELTAS:
            movement_failed = repeated_percept and percept['wall_ahead']
            if not movement_failed:
                self.position = self._next_cell(self.last_action)
                self.visited_cells.add(self.position)
            self.facing = self.last_action

        if percept['wall_ahead']:
            self.blocked_cells.add(self._next_cell(self.facing))

        if repeated_percept and percept['wall_ahead'] and self.last_action in self.DELTAS:
            self.tried_actions.add(self.last_action)
        else:
            self.tried_actions.clear()

        self.percept_history.append(dict(percept))
        return repeated_percept

    def _choose_alternative(self) -> str:
        unvisited = [
            action for action in self.ACTIONS
            if action not in self.tried_actions
            and self._next_cell(action) not in self.blocked_cells
            and self._next_cell(action) not in self.visited_cells
        ]
        if unvisited:
            return unvisited[0]

        available = [
            action for action in self.ACTIONS
            if action not in self.tried_actions
            and self._next_cell(action) not in self.blocked_cells
        ]
        return available[0] if available else 'Right'

    def sense_and_act(self, percept: dict) -> str:
        self._update_state(percept)

        if percept['food_here']:
            action = 'Suck'
        else:
            forward_cell = self._next_cell(self.facing)
            if percept['wall_ahead'] or forward_cell in self.visited_cells:
                action = self._choose_alternative()
            else:
                action = self.facing

        self.last_percept = dict(percept)
        self.last_action = action
        self.action_history.append(action)
        return action
