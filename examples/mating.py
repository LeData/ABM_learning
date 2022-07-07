import random

from math import tan, pi
import numpy as np
from mesa import Model, Agent
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from collections import defaultdict, Counter
from functools import partial


class DNA:

    def __init__(self, sex, active_interest_threshold, passive_interest_threshold, appearance, preference):
        self.sex = sex
        self.ait = active_interest_threshold
        self.pit = passive_interest_threshold
        self.appearance = appearance
        self.preference = preference

    def __repr__(self):
        return "".join([self.sex, self.ait, self.pit, self.appearance, self.preference])

    @classmethod
    def generate_randomly(cls, sex, active_interest_threshold, passive_interest_threshold):
        preference = 2 * np.random.random(5) - 1
        appearance = np.random.random(5)
        return cls(sex=sex,
                   active_interest_threshold=active_interest_threshold,
                   passive_interest_threshold=passive_interest_threshold,
                   appearance=appearance,
                   preference=preference)


class MatingA(Agent):

    min_relationship = 30  # days

    def __init__(self, sex, ait: float, pit: float, preference, appearance, unique_id, model: "MatingModel"):
        """

        :param sex:
        :param ait: active interest threshold (between 0 and 1)
        :param pit: passive interest threshold (between 0 and 1)
        :param preference: preferences for partners (array of length n between -1 and 1)
        :param appearance: appearance to other agents (array of length n between 0 and 1)
        :param unique_id: identifier
        :param model: model the agent evolves in.
        """
        super().__init__(unique_id=unique_id, model=model)

        # individual properties
        self._preference = preference
        self._appearance = appearance
        if len(self.preference)!= len(self.appearance):
            raise AssertionError("preferences and appearance should have the same size")
        self._n = len(self.preference)
        self._sex = sex
        self.ait = ait
        self.pit = pit

        # pairing properties
        self.paired = False
        self.proposals = tuple()
        self.pairs = tuple()
        self.separation_countdown = None
        self.matching_likelihood = defaultdict(float)
        self.rejected_by = set()
        self.rejected = set()

    @classmethod
    def from_dna(cls, dna: DNA, unique_id, model: "MatingModel"):
        return cls(sex=dna.sex, ait=dna.ait, pit=dna.pit,
                   preference=dna.preference, appearance=dna.appearance, unique_id=unique_id, model=model)

    @property
    def sex(self):
        return self._sex

    @property
    def active_threshold(self):
        return self.ait

    @property
    def passive_threshold(self):
        return self.pit

    @property
    def appearance(self):
        return self._appearance

    @property
    def preference(self):
        return self._preference

    @property
    def separations(self):
        return self.pairs[:-1] if self.paired else self.pairs

    @property
    def is_incel(self):
        if not self.paired:
            accessible_market = sum(self.model.single_counts[sex] for sex in self.model.market_size
                                    if sex != self.sex)
            return accessible_market <= len(self.rejected_by) + len(self.rejected) + len(self.separations)
        else:
            return False

    def propose_relationship(self, agent):
        self.proposals = tuple([*self.proposals, agent])
        result = agent.respond_to_proposal(self)
        if result:
            self.start_relationship_with(agent)
            agent.start_relationship_with(self)
        else:
            self._appearance += self.evaluate_credit(agent) * self.evaluate_inadequacy(agent)
            self.rejected_by.add(agent)
            agent.rejected.add(self)

    def respond_to_proposal(self, agent):
        return self.evaluate_interest(agent=agent, active=False) >= self.passive_threshold

    def start_relationship_with(self, agent):
        self.paired = True
        self.pairs = tuple([*self.pairs, agent])
        compatibility = self.evaluate_compatibility(agent)
        self.separation_countdown = self.min_relationship * max(1, tan(pi*compatibility/2)**4)

    def end_relationship(self):
        self.paired = False

    def evaluate_compatibility(self, agent):
        return np.dot(agent.appearance, self.preference) / self._n

    def evaluate_interest(self, agent: "MatingA", active: bool = True):
        total_interest = self.matching_likelihood[agent.unique_id] + self.evaluate_compatibility(agent)
        if active:
            self.matching_likelihood[agent.unique_id] = total_interest
        return total_interest

    def evaluate_credit(self, agent: "MatingA"):
        return .1

    def evaluate_inadequacy(self, agent):
        return agent.preference - self.appearance

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
                          if a != self and a.sex != self.sex}  # Agents in the neighborhood
            disqualified = self.rejected_by.union(set(self.separations))
            for agent in nbd_agents.difference(disqualified):
                if self.evaluate_interest(agent, active=True) >= self.active_threshold:
                    self.propose_relationship(agent)
                    if self.paired:
                        break


# noinspection PyPep8Naming
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
            sex = "M" if i < balance * num_agents else "F"
            it = interest_thresholds[sex]
            dna = DNA.generate_randomly(sex=sex,
                                        active_interest_threshold=it,
                                        passive_interest_threshold=it)
            agent = MatingA.from_dna(dna=dna, unique_id=i, model=self)
            pos = tuple(self.random.randrange(p) for p in (self.grid.width, self.grid.height))
            self.schedule.add(agent)
            self.grid.place_agent(agent, pos=pos)

        self.datacollector = DataCollector(model_reporters={
                                                **{f"Incel_{a_t}": partial(compute_incel_prop, sex=a_t)
                                                   for a_t in self.types},
                                                "Incel": compute_incel_prop,
                                                **{f"Single_{a_t}": partial(compute_singles_prop,sex=a_t)
                                                   for a_t in self.types},
                                                "Single": compute_singles_prop,
                                                **{f"Avg_rej_{a_t}": partial(compute_singles_rejections, sex=a_t)
                                                   for a_t in self.types},
                                                "Avg_rej": compute_singles_rejections
                                                },
                                           agent_reporters={
                                               #"Preference": "preference",
                                               #"Pairs": lambda x: [a.unique_id for a in x.pairs],
                                               #"Rejections": lambda x: [a.unique_id for a in x.rejected_by],
                                               #"Separations": lambda x: [a.unique_id for a in x.separations]
                                               })

        self._market_size = Counter([a.sex for a in self.schedule.agents])
        self.running = True

    @property
    def market_size(self):
        return self._market_size

    @property
    def incel_counts(self):
        return Counter([agent.sex for agent in self.schedule.agents if agent.is_incel])

    @property
    def single_counts(self):
        return Counter([agent.sex for agent in self.schedule.agents if not agent.paired])

    @property
    def average_rejections(self):
        rejections = defaultdict(list)
        for agent in self.schedule.agents:
            if not agent.paired:
                rejections[agent.sex].append(len(agent.rejected_by))
        return {a_t: np.mean(x) for a_t, x in rejections.items()}

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)


def compute_incel_prop(model: MatingModel, sex: str = None):
    incels = model.incel_counts
    singles = model.single_counts
    if sex:
        if singles[sex] > 0:
            return incels[sex] / singles[sex]
        else:
            return 0
    total_singles = sum(singles.values())
    if total_singles > 0:
        return sum(incels.values()) / total_singles
    else:
        return 0


def compute_singles_prop(model: MatingModel, sex: str = None):
    singles = model.single_counts
    totals = model.market_size
    if sex:
        return singles[sex] / totals[sex]
    else:
        return sum(singles.values()) / sum(totals.values())


def compute_singles_rejections(model: MatingModel, sex: str = None):
    if sex:
        return model.average_rejections[sex]
    else:
        singles = model.single_counts
        n_singles = sum(singles.values())
        if n_singles > 0:
            return sum([singles[a_t] * model.average_rejections[a_t] for a_t in model.average_rejections]) / n_singles
        else:
            return 0
