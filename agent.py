import heapq
from collections import deque

from utils import Action, ActionType, Condition


class Agent:
    def __init__(self, game):
        """
        Initializes the agent with the given game instance.

        Args:
            game: An instance of the game that the agent will interact with.
        """
        self.game = game
        self.rows = game.rows
        self.cols = game.cols

    def play(self):
        """
        Executes the game loop for the agent.

        The agent continuously observes the game state, determines the next action,
        and performs the action until the game reaches a terminal condition.

        Returns:
            goal_test (Condition): The final state of the game, indicating whether
                                   the game is still in progress, won, or reveal a bomb.
        """
        raise NotImplementedError()

    def get_neighbors(self, x, y):
        """
        Get the neighboring coordinates of a given cell in a board.

        Args:
            x (int): The x-coordinate of the cell.
            y (int): The y-coordinate of the cell.

        Returns:
            list of tuple: A list of tuples representing the coordinates of the neighboring cells.
                           Only includes neighbors that are within the bounds of the board.
        """
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        return [(x + dx, y + dy) for dx, dy in directions if 0 <= x + dx < self.game.rows and 0 <= y + dy < self.game.cols]

    def state_to_key(self, state):
        """
        Helper method to convert the 2D state list into a hashable tuple of tuples.
        """
        return tuple(tuple(row) for row in state)


class ManualGuiAgent(Agent):
    def __init__(self, game):
        super().__init__(game)

    def play(self):
        pass


class SearchAgent(Agent):
    def __init__(self, game, clues):
        super().__init__(game)
        self.clues = clues

    def is_goal_state(self, clue, state):
        """
        Determines if the given state satisfies the goal condition based on the provided clues.

        Args:
            clue (list of list of int): A 2D list representing the clues for each cell. 
                                        A clue is an integer indicating the number of bombs in the neighboring cells.
            state (list of list of str): A 2D list representing the current state of the board in search. 
                                         Each cell can be 'B' for bomb or 'S' for safe (i.e., non-bomb).

        Returns:
            bool: True if the state satisfies all the clues, False otherwise.
        """
        for i in range(self.rows):
            for j in range(self.cols):
                if isinstance(clue[i][j], int)  and clue[i][j] >= 0:  # If it's a clue
                    neighbors = self.get_neighbors(i, j)
                    bomb_count = sum(1 for ni, nj in neighbors if state[ni][nj] == 'B')
                    if bomb_count != clue[i][j]:
                        return False
        return True

    def get_next_actions(self, state):
        """
        Generate the list of possible next actions based on the current state.

        Args:
            state (list of list of str): The current state of the board, represented as a 2D list.
                                         Each element can be 'B', 'S', or '_'.
                                         B: bomb, S: safe, _: unknown.

        Returns:
            list of tuple: A list of possible actions. Each action is represented as a tuple (i, j, label),
                           where 'i' is the row index, 'j' is the column index, and 'label' is either 'B' or 'S'.
        """
        next_actions = []
        for i in range(self.rows):
            for j in range(self.cols):
                if state[i][j] == '_':
                    next_actions.append((i, j, 'B'))
                    next_actions.append((i, j, 'S'))
        return next_actions

    def search(self, clues):
        """
        Perform a search based on the provided clues.

        Args:
            clues (list): A list of clues to guide the search.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError()
    
    def print_solution(self, state, actions):
        """
        Prints the solution to the search problem.

        Args:
            state (list of list of str): The final state of the problem, represented as a 2D list.
            actions (list of tuples): The sequence of actions taken to reach the final state.

        Prints:
            The solution details including the action sequence and the final state.
        """
        print("Found a solution")
        print("Action sequence:", actions)
        print("Final state:")
        for row in state:
            print(' '.join(row))
        print()

    def reveal_initial_clues(self):
        """
        Reveals the initial clues on the board.

        Iterates through the clues and reveals each clue that is an integer.

        Returns:
            obs: The observation after the last reveal action.
            goal_test: The result of the goal test after the last reveal action.
        """
        for x in range(len(self.clues)):
            for y in range(len(self.clues[0])):
                clue = self.clues[x][y]
                if isinstance(clue, int):
                    obs, goal_test = self.game.step(Action(ActionType.REVEAL, x, y))
        return obs, goal_test

    def play(self):
        """
        Executes the agent's logic based on a search algorithm.

        The method first reveals all safe cells (i.e., cells with integer clues).
        Then, it continues with a search algorithm to determine the next actions
        (either flagging or revealing cells) until the game is no longer in progress.

        Raises:
            ValueError: If no solution is found during the search.

        Returns:
            Condition: The final state of the game.
        """
        # Reveal the clues first if it is a safe cell, i.e., integer
        obs, goal_test = self.reveal_initial_clues()

        # Continue the search algorithm
        while True:
            state, _ = self.search(obs)
            if state is None:
                raise ValueError("No solution found.")
            for x in range(self.rows):
                for y in range(self.cols):
                    if state[x][y] == 'B':
                        obs, goal_test = self.game.step(Action(ActionType.FLAG, x, y))
                    else:
                        obs, goal_test = self.game.step(Action(ActionType.REVEAL, x, y))
                    if goal_test != Condition.IN_PROGRESS:
                        return goal_test


class BFSAgent(SearchAgent):
    def __init__(self, game, clues):
        super().__init__(game, clues)

    def search(self, clues):
        initial_state = [['_' for _ in range(self.cols)] for _ in range(self.rows)]  # Start with all cells unknown

        # Initialize BFS state with all clue cells marked as safe
        for i in range(self.rows):
            for j in range(self.cols):
                if isinstance(clues[i][j], int) and clues[i][j] >= 0:  # Clue cells
                    initial_state[i][j] = 'S'

        # TODO: Complete the BFS search algorithm
        # Create an initial state: mark clue cells as safe, rest as unknown ('_')
        initial_state = [['_' for _ in range(self.cols)] for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(self.cols):
                if isinstance(clues[i][j], int) and clues[i][j] >= 0:
                    initial_state[i][j] = 'S'

        # Use a FIFO queue for BFS: each node is (state, action_sequence)
        queue = deque()
        queue.append((initial_state, []))
        visited = set()
        visited.add(self.state_to_key(initial_state))

        while queue:
            state, actions = queue.popleft()
            if self.is_goal_state(clues, state):
                return state, actions
            for action in self.get_next_actions(state):
                i, j, label = action
                if state[i][j] != '_':  # Skip if already set
                    continue
                # Generate a new state with this action applied
                new_state = [row[:] for row in state]
                new_state[i][j] = label
                key = self.state_to_key(new_state)
                if key not in visited:
                    visited.add(key)
                    new_actions = actions + [action]
                    queue.append((new_state, new_actions))
        return None, []

    

class DFSAgent(SearchAgent):
    def __init__(self, game, clues):
        super().__init__(game, clues)

    def search(self, clues):
        initial_state = [['_' for _ in range(self.cols)] for _ in range(self.rows)]

        # Initialize DFS state with all clue cells marked as safe
        for i in range(self.rows):
            for j in range(self.cols):
                if isinstance(clues[i][j], int) and clues[i][j] >= 0:  # Clue cells
                    initial_state[i][j] = 'S'

        # TODO: Complete the DFS search algorithm
        # Use a LIFO stack for DFS: each node is (state, action_sequence)
        stack = []
        stack.append((initial_state, []))
        visited = set()
        visited.add(self.state_to_key(initial_state))

        while stack:
            state, actions = stack.pop()
            if self.is_goal_state(clues, state):
                return state, actions
            for action in self.get_next_actions(state):
                i, j, label = action
                if state[i][j] != '_':  # Already assigned
                    continue
                new_state = [row[:] for row in state]
                new_state[i][j] = label
                key = self.state_to_key(new_state)
                if key not in visited:
                    visited.add(key)
                    new_actions = actions + [action]
                    stack.append((new_state, new_actions))
        return None, []



class AStarAgent(SearchAgent):
    def __init__(self, game, clues):
        super().__init__(game, clues)

    def heuristic(self, clues, state):
        """
        Calculate the heuristic value for a given state based on the provided clues.

        Args:
            clues (list of list of int): A 2D list representing the clues for each cell. 
                                         Clues are integers where a non-negative value indicates 
                                         the number of bombs in the neighboring cells.
            state (list of list of str): A 2D list representing the current state of the board. 
                                         Each cell can be 'B' for bomb or any other value for non-bomb.

        Returns:
            int: The heuristic value for the given state. A higher heuristic value indicates a less 
                 desirable state. The heuristic value is calculated based on the difference between 
                 the number of bombs around a clue cell and the clue value. A high heuristic value 
                 (1000 or more) is added for invalid states where the number of bombs exceeds the clue value.
        """
        # TODO: Complete the heuristic function
        h = 0
        for i in range(self.game.rows):
            for j in range(self.game.cols):
                if isinstance(clues[i][j], int) and clues[i][j] >= 0:
                    bomb_count = 0
                    for ni, nj in self.get_neighbors(i, j):
                        if state[ni][nj] == 'B':
                            bomb_count += 1
                    if bomb_count > clues[i][j]:
                        # Over-assigned bombs – penalize heavily.
                        return 1000 + (bomb_count - clues[i][j])
                    else:
                        # Under-assigned bombs contribute to the heuristic.
                        h += (clues[i][j] - bomb_count)
        return h

    def search(self, clues):
        initial_state = [['_' for _ in range(self.game.cols)] for _ in range(self.game.rows)]

        # Initialize A* state with all clue cells marked as safe
        for i in range(self.game.rows):
            for j in range(self.game.cols):
                if isinstance(clues[i][j], int) and clues[i][j] >= 0:  # Clue cells
                    initial_state[i][j] = 'S'

        # TODO: Complete the BFS search algorithm
        # A* search: each node is (priority, cost, state, action_sequence)
        open_set = []
        start_cost = 0
        start_priority = start_cost + self.heuristic(clues, initial_state)
        heapq.heappush(open_set, (start_priority, start_cost, initial_state, []))
        # Use a dictionary to store the minimum cost found for a given state key.
        visited = {self.state_to_key(initial_state): start_cost}

        while open_set:
            priority, cost, state, actions = heapq.heappop(open_set)
            if self.is_goal_state(clues, state):
                return state, actions
            for action in self.get_next_actions(state):
                i, j, label = action
                if state[i][j] != '_':
                    continue
                new_state = [row[:] for row in state]
                new_state[i][j] = label
                new_cost = cost + 1
                key = self.state_to_key(new_state)
                if key not in visited or new_cost < visited[key]:
                    visited[key] = new_cost
                    h = self.heuristic(clues, new_state)
                    new_priority = new_cost + h
                    new_actions = actions + [action]
                    heapq.heappush(open_set, (new_priority, new_cost, new_state, new_actions))
        return None, []
