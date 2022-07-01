from mesa import Model, Agent
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from mesa.space import MultiGrid


def compute_gini(model):
    agent_wealths = [agent.wealth for agent in model.schedule.agents]
    x = sorted(agent_wealths)
    N = model.num_agents
    B = sum(xi * (N - i) for i, xi in enumerate(x)) / (N * sum(x))
    return 1 + (1 / N) - 2 * B


class MoneyAgent(Agent):

    def __init__(self, unique_id, model: Model):
        super().__init__(unique_id=unique_id, model=model)
        self._wealth = 1

    @property
    def wealth(self):
        return self._wealth

    def give_to(self, agent, amount: int):
        if self.wealth > 0:
            self._wealth -= amount
            agent.receive(amount)

    def receive(self, amount: int):
        if amount < 0:
            raise ValueError("The amount must be positive.")
        self._wealth += amount

    def move(self):
        if self.wealth > 0:
            possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
            new_position = self.random.choice(possible_steps)
            self.model.grid.move_agent(self, new_position)

    def step(self):
        """
        Action the agent takes
        :return: None
        """
        self.move()
        amount = 1
        potential_receivers = self.model.grid.get_cell_list_contents([self.pos])
        if len(potential_receivers) > 0:
            receiver = self.random.choice(potential_receivers)
            self.give_to(receiver, amount)


class MoneyModel(Model):

    def __init__(self, n: int, width, height):

        self.num_agents = n
        self.grid = MultiGrid(width=width, height=height, torus=True)
        self.schedule = RandomActivation(self)

        for i in range(self.num_agents):
            a = MoneyAgent(i, self)
            self.schedule.add(a)
            pos = tuple(self.random.randrange(p) for p in (self.grid.width, self.grid.height))
            self.grid.place_agent(a, pos=pos)  # This modifies the agent's x,y property directly

        self.datacollector = DataCollector(model_reporters={"Gini": compute_gini}, agent_reporters={"Wealth": "wealth",
                                                                                                    "Pos": "pos"})
        self.running = True

    def __repr__(self):
        return " ".join([str(a.wealth) for a in self.schedule.agents])

    def step(self):
        """
        Action the model takes at every step
        :return:
        """
        self.schedule.step()
        self.datacollector.collect(self)
