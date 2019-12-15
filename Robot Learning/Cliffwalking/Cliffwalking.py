ACTIONS = ['up', 'right', 'down', 'left', 'start']


class State:
    def __init__(self, x, y, reward, isCliff):
        self.x = x
        self.y = y
        self.reward = reward
        self.isCliff = isCliff

    def __eq__(self, other):
        isEqual = self.x == other.x and self.y == other.y
        return isEqual


class Grid:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.grid = self.init_grid()
        self.set_start_state(0, 0)
        self.set_goal_state(col - 1, 0)

    def set_start_state(self, x, y):
        self.start_state = self.grid[y][x]

    def set_goal_state(self, x, y):
        self.goal_state = self.grid[y][x]

    def print_path(self, path):
        grid = []
        for y in range(self.row):
            columns = []
            for x in range(self.col):
                if (0 < x) and (x < self.col - 1) and (y == 0):
                    columns = columns + ['C']
                else:
                    columns = columns + [0]
            grid.append(columns)

        for state in path:
            grid[state.y][state.x] = 1

        lines = ''
        for y in reversed(range(self.row)):
            for x in range(self.col):
                lines += str(grid[y][x]) + ' '
            lines = lines + '\n'

        print(lines)

    def init_grid(self):
        grid = []
        for y in range(self.row):
            columns = []
            for x in range(self.col):
                isCliff = False
                if (0 < x) and (x < self.col - 1) and (y == 0):
                    rew = -100
                    isCliff = True
                else:
                    rew = -1
                columns = columns + [State(x, y, rew, isCliff)]
            grid.append(columns)
        return grid

    def init_q_values(self):
        q_values = []
        for y in range(self.row):
            columns = []
            for x in range(self.col):
                q_value = dict()
                for action in ACTIONS:
                    if self.take_action(self.grid[y][x], action):
                        q_value[action] = 0.0
                columns = columns + [q_value]
            q_values.append(columns)
        return q_values

    def take_action(self, state, action):
        if self.grid[state.y][state.x].isCliff:
            if action is 'start':
                return self.start_state
            else:
                return
        new_x = -1
        new_y = -1
        if action == 'up':
            new_x = state.x
            new_y = state.y + 1
        elif action == 'right':
            new_x = state.x + 1
            new_y = state.y
        elif action == 'down':
            new_x = state.x
            new_y = state.y - 1
        elif action == 'left':
            new_x = state.x - 1
            new_y = state.y

        if (0 <= new_x) and (new_x < self.col) and (0 <= new_y) and (new_y < self.row):
            return self.grid[new_y][new_x]
