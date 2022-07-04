import random

from math import log
import numpy as np
from mesa import Model, Agent
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from collections import defaultdict, Counter
from functools import partial


class MatingA(Agent):

    def __init__(self, interest_threshold, preference: np.array, appearance: np.array, agent_type, unique_id,
                 model: Model):
        super().__init__(unique_id=unique_id, model=model)

        # individual properties
        self.type = agent_type
        self._preference = preference
        self._appearance = appearance
        self.active_threshold = interest_threshold
        self.passive_threshold = interest_threshold

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
        return self._appearance

    @property
    def preference(self):
        return self._preference

    @property
    def is_incel(self):
        accessible_market = sum(self.model.single_counts[a_type] for a_type in self.model.market_size
                                if a_type != self.type)
        return accessible_market <= .8 * len(self.rejected_by)

    def propose_relationship(self, agent):
        self.proposals = tuple([*self.proposals, agent])
        result = agent.evaluate_interest(self, active=False)
        if result:
            self.start_relationship_with(agent)
            agent.start_relationship_with(self)
        else:
            self._appearance += self.evaluate_credit(agent) * self.evaluate_inadequacy(agent)  # Make appearance change, not preference.
            self.rejected_by.add(agent)

    def start_relationship_with(self, agent):
        self.paired = True
        self.pairs = tuple([*self.pairs, agent])
        interest = np.dot(agent.appearance, self.preference)
        self.separation_countdown = min(10, interest**10)

    def end_relationship(self):
        self.paired = False
        self.separations = tuple([*self.separations, self.pairs[-1]])

    def evaluate_interest(self, agent: Agent, active: bool = True):
        interest = np.dot(agent.appearance, self.preference)
        if active:
            self.matching_likelihood[agent.unique_id] += interest
            return self.matching_likelihood[agent.unique_id]
        else:
            interest += self.matching_likelihood[agent.unique_id]
            return interest > self.passive_threshold

    def evaluate_differences(self, appearance):
        return self.active_threshold - appearance

    def evaluate_credit(self, agent):
        return .1

    def evaluate_inadequacy(self, agent):
        return agent.evaluate_differences(self.appearance)

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def step(self):
        if self.paired:
            self.separation_countdown -= 1
            if self.separation_countdown <= 0:
                self.end_relationship()
                self.pairs[-1].end_relationship()
        else:
            if not self.is_incel:
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



class MatingModel(Model):
    types = ["M", "F"]

    def __init__(self, num_agents, balance, it_M, it_F, width, height):
        interest_thresholds = {
            "M": it_M,
            "F": it_F
        }
        self.num_agents = num_agents
        if balance not in [0, 1]:
            self.balance = balance
        else:
            raise ValueError("the balance cannot be 0 or 1")

        self.grid = MultiGrid(width=width, height=height, torus=True)
        self.schedule = RandomActivation(self)

        for i in range(num_agents):
            agent_type = "M" if random.random() < balance else "F"
            a = MatingA(interest_threshold=interest_thresholds[agent_type],
                        preference=2 * np.random.random(5) - 1,
                        appearance=np.random.random(5),
                        agent_type=agent_type, unique_id=i, model=self)
            pos = tuple(self.random.randrange(p) for p in (self.grid.width, self.grid.height))
            self.schedule.add(a)
            self.grid.place_agent(a, pos=pos)

        self.datacollector = DataCollector(model_reporters={**{f"Incel_{a_t}": partial(compute_incels_prop,
                                                                                       agent_type=a_t)
                                                               for a_t in self.types},
                                                            "Incel": compute_incels_prop,
                                                            **{f"Single_{a_t}": partial(compute_singles_prop,
                                                                                        agent_type=a_t)
                                                               for a_t in self.types},
                                                            "Single": compute_singles_prop,
                                                            **{f"Avg_rej_{a_t}": partial(compute_singles_rejections,
                                                                                         agent_type=a_t)
                                                             for a_t in self.types},
                                                            "Avg_rej": compute_singles_rejections
                                                            },
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

    @property
    def incel_counts(self):
        return Counter([agent.type for agent in self.schedule.agents if agent.is_incel])

    @property
    def single_counts(self):
        return Counter([agent.type for agent in self.schedule.agents if not agent.paired])

    @property
    def average_rejections(self):
        rejections = defaultdict(list)
        for agent in self.schedule.agents:
            if not agent.paired:
                rejections[agent.type].append(len(agent.rejected_by))
        return {a_t: np.mean(x) for a_t, x in rejections.items()}

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)


def compute_incels_prop(model: MatingModel, agent_type: str = None):
    incels = model.incel_counts
    singles = model.single_counts
    if agent_type:
        if singles[agent_type] > 0:
            return incels[agent_type] / singles[agent_type]
        else:
            return 0
    total_singles = sum(singles.values())
    if total_singles > 0:
        return sum(incels.values()) / total_singles
    else:
        return 0


def compute_singles_prop(model: MatingModel, agent_type: str = None):
    singles = model.single_counts
    totals = model.market_size
    if agent_type:
        return singles[agent_type] / totals[agent_type]
    else:
        return sum(singles.values()) / sum(totals.values())


def compute_singles_rejections(model: MatingModel, agent_type: str = None):
    if agent_type:
        return model.average_rejections[agent_type]
    else:
        singles = model.single_counts
        n_singles = sum(singles.values())
        if n_singles > 0:
            return sum([singles[a_t] * model.average_rejections[a_t] for a_t in model.average_rejections]) / n_singles
        else:
            return 0
