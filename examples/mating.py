import random

from math import log
import numpy as np
from mesa import Model, Agent
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from collections import defaultdict, Counter


interest_thresholds = {
    "female": 2,
    "male": 1.5
}


class MatingA(Agent):

    def __init__(self, interest_threshold, preference: np.array, agent_type, unique_id, model: Model):
        super().__init__(unique_id=unique_id, model=model)

        # individual properties
        self.type = agent_type
        self.preference = preference
        self.interest_threshold = interest_threshold

        # pairing properties
        self.paired = False
        self.proposals = tuple()
        self.pairs = tuple()
        self.separations = tuple()
        self.separation_countdown = None
        self.matching_likelihood = defaultdict(float)
        self.rejected_by = set()

    @property
    def appearance(self):
        return self.preference

    @property
    def is_incel(self):
        accessible_market = sum(self.model.market_size[a_type] for a_type in self.model.market_size
                                if a_type != self.type)
        return accessible_market <= len(self.rejected_by)

    def propose_relationship(self, agent):
        self.proposals = tuple([*self.proposals, agent.unique_id])
        result = agent.evaluate_interest(self)
        if result:
            self.start_relationship_with(agent)
            agent.start_relationship_with(self)
        else:
            self.preference += self.evaluate_inadequacy(agent)
            self.rejected_by.add(agent)

    def start_relationship_with(self, agent):
        self.paired = True
        self.pairs = tuple([*self.pairs, agent])
        interest = np.dot(agent.appearance, self.preference)
        self.separation_countdown = min(10, interest**10)

    def end_relationship(self):
        self.paired = False
        self.separations = tuple([*self.separations, self.pairs[-1]])

    def evaluate_interest(self, agent: Agent):
        interest = np.dot(agent.appearance, self.preference)
        self.matching_likelihood[agent.unique_id] += interest
        return self.matching_likelihood[agent.unique_id] > self.interest_threshold

    def evaluate_differences(self, appearance):
        return self.interest_threshold - appearance

    def evaluate_credit(self, agent):
        return .1

    def evaluate_inadequacy(self, agent):
        credit_to_give = self.evaluate_credit(agent)
        return credit_to_give * agent.evaluate_differences(self.appearance)

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def step(self):
        if self.paired:
            self.separation_countdown -= 1
            if self.separation_countdown <= 0:
                self.end_relationship()
        else:
            self.move()
            nbd_cells = self.model.grid.get_neighborhood(self.pos, moore=False, include_center=True)
            nbd_agents = {a for a in self.model.grid.get_cell_list_contents(nbd_cells)
                          if a != self and a.type != self.type}  # Agents in the neighborhood
            potential_matches = nbd_agents.difference(self.rejected_by.union(set(self.separations)))
            for agent in potential_matches:
                if self.evaluate_interest(agent):
                    self.propose_relationship(agent)
                    if self.paired:
                        break


def compute_incels_prop(model: Model):
    incels = [agent for agent in model.schedule.agents if agent.is_incel]
    return len(incels) / model.num_agents


class MatingModel(Model):

    def __init__(self, num_agents, balance, width, height):
        self.num_agents = num_agents
        self.balance = balance

        self.grid = MultiGrid(width=width, height=height, torus=True)
        self.schedule = RandomActivation(self)

        for i in range(num_agents):
            agent_type = "male" if random.random() < balance else "female"
            a = MatingA(interest_threshold=interest_thresholds[agent_type],
                        preference=2 * np.random.random(5) - 1, agent_type=agent_type, unique_id=i, model=self)
            pos = tuple(self.random.randrange(p) for p in (self.grid.width, self.grid.height))
            self.schedule.add(a)
            self.grid.place_agent(a, pos=pos)

        self.datacollector = DataCollector(model_reporters={"Incel_prop": compute_incels_prop},
                                           agent_reporters={"Preference": "preference",
                                                            "Pairs": lambda x: [a.unique_id for a in x.pairs],
                                                            "Rejections": lambda x: [a.unique_id for a in x.rejected_by],
                                                            "Separations": lambda x: [a.unique_id for a in x.separations]
                                                            })

        self._market_size = Counter([agent.type for agent in self.schedule.agents])
        self.running = True

    @property
    def market_size(self):
        return self._market_size

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)
