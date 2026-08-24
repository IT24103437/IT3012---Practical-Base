# agent.py
import heapq
import math
from collections import deque


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


class SearchAgent:
    """Find offline plans using uninformed graph-search algorithms."""

    DELTAS = {
        'Up': (0, 1),
        'Down': (0, -1),
        'Left': (-1, 0),
        'Right': (1, 0)
    }

    def __init__(self):
        self.plan = []
        self.active_algo = 'AStar'
        self.current_pos = (0, 0)

    def manhattan_distance(self, pos, goal) -> int:
        """Return the four-way grid distance from pos to goal."""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def euclidean_distance(self, pos, goal) -> float:
        """Return the straight-line distance from pos to goal."""
        return math.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)

    def sense_and_act(self, percept: dict) -> str:
        if not self.plan:
            self.plan = self.make_plan(percept)

        if not self.plan:
            return 'Stay'

        action = self.plan.pop(0)
        dx, dy = self.DELTAS[action]
        self.current_pos = self.current_pos[0] + dx, self.current_pos[1] + dy
        return action

    def make_plan(self, percept: dict) -> list:
        """Build a complete route to the closest reachable food pellet."""
        food_positions = [tuple(position) for position in percept['all_food']]
        if not food_positions:
            return []

        if self.active_algo == 'BFS':
            search = self.bfs_search
        elif self.active_algo == 'DFS':
            search = self.dfs_search
        elif self.active_algo == 'UCS':
            search = self.ucs_search
        elif self.active_algo == 'AStar':
            search = self.astar_search
        else:
            raise ValueError(f"Unknown search algorithm: {self.active_algo}")

        ordered_goals = sorted(
            food_positions,
            key=lambda goal: abs(goal[0] - self.current_pos[0]) + abs(goal[1] - self.current_pos[1])
        )

        for goal in ordered_goals:
            plan = search(self.current_pos, goal, percept['walls'], percept['grid_size'])
            if plan or goal == self.current_pos:
                return plan

        return []

    def expand(self, state, walls, grid_size):
        """Yield every legal action, successor state, and step cost."""
        width, height = grid_size
        x, y = state

        for action, (dx, dy) in self.DELTAS.items():
            next_state = (x + dx, y + dy)
            inside_grid = 0 <= next_state[0] < width and 0 <= next_state[1] < height
            if inside_grid and next_state not in walls:
                yield action, next_state, 1

    def bfs_search(self, start, goal, walls, grid_size) -> list:
        """Use a FIFO frontier to find the shallowest path."""
        walls = set(walls)
        frontier = deque([(start, [])])
        reached = {start}

        while frontier:
            state, path = frontier.popleft()
            if state == goal:
                return path

            for action, next_state, _ in self.expand(state, walls, grid_size):
                if next_state not in reached:
                    reached.add(next_state)
                    frontier.append((next_state, path + [action]))

        return []

    def dfs_search(self, start, goal, walls, grid_size) -> list:
        """Use a LIFO frontier to explore the deepest path first."""
        walls = set(walls)
        frontier = [(start, [])]
        reached = {start}

        while frontier:
            state, path = frontier.pop()
            if state == goal:
                return path

            for action, next_state, _ in self.expand(state, walls, grid_size):
                if next_state not in reached:
                    reached.add(next_state)
                    frontier.append((next_state, path + [action]))

        return []

    def ucs_search(self, start, goal, walls, grid_size) -> list:
        """Use a priority frontier ordered by total path cost g(n)."""
        walls = set(walls)
        frontier = [(0, 0, start, [])]
        best_cost = {start: 0}
        reached = set()
        tie_breaker = 0

        while frontier:
            cost, _, state, path = heapq.heappop(frontier)
            if state in reached:
                continue
            reached.add(state)

            if state == goal:
                return path

            for action, next_state, step_cost in self.expand(state, walls, grid_size):
                new_cost = cost + step_cost
                if next_state not in reached and new_cost < best_cost.get(next_state, float('inf')):
                    best_cost[next_state] = new_cost
                    tie_breaker += 1
                    heapq.heappush(frontier, (new_cost, tie_breaker, next_state, path + [action]))

        return []

    def astar_search(self, start_pos, goal_pos, walls, grid_size, heuristic_type='manhattan') -> list:
        """Find a path using the combined priority f(n) = g(n) + h(n)."""
        heuristic_name = heuristic_type.lower()
        if heuristic_name == 'manhattan':
            heuristic = self.manhattan_distance
        elif heuristic_name == 'euclidean':
            heuristic = self.euclidean_distance
        else:
            raise ValueError("heuristic_type must be 'manhattan' or 'euclidean'")

        walls = set(walls)
        reached_states = set()
        best_cost = {start_pos: 0}

        start_g = 0
        start_h = heuristic(start_pos, goal_pos)
        frontier = [(start_g + start_h, start_g, start_pos, [])]

        while frontier:
            _, current_g, current_pos, path_taken = heapq.heappop(frontier)
            if current_pos in reached_states:
                continue

            if current_pos == goal_pos:
                return path_taken

            reached_states.add(current_pos)

            for action, neighbor, step_cost in self.expand(current_pos, walls, grid_size):
                if neighbor in reached_states:
                    continue

                new_g = current_g + step_cost
                if new_g < best_cost.get(neighbor, float('inf')):
                    best_cost[neighbor] = new_g
                    new_h = heuristic(neighbor, goal_pos)
                    new_f = new_g + new_h
                    heapq.heappush(
                        frontier,
                        (new_f, new_g, neighbor, path_taken + [action])
                    )

        return []
