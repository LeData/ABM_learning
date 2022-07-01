from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.UserParam import UserSettableParameter

from boltzman_money import MoneyModel


class GiniElement(TextElement):
    def render(self, model):
        try:
            value = model.datacollector.get_model_vars_dataframe()['Gini'].values[-1]
        except IndexError:
            value = None
        return f"The Gini Coefficient is {value}"


def MoneyDraw(agent):
    if agent:
        return {
            "Shape": "circle",
            "r": 1 + 0.5 * agent.wealth,
            "Filled": True,
            "Layer": 0,
            "Color": "red"
        }


if __name__ == "__main__":
    height, width = (50, 10)
    g_element = GiniElement()
    canvas_element = CanvasGrid(MoneyDraw, height, width, 500, 500)
    gini_chart = ChartModule([{"Label": "Gini", "Color": "Black"}])

    model_params = {
        "height": height,
        "width": width,
        "n": UserSettableParameter("slider", "agent count", 20, 0, 100, step=1)
    }
    server = ModularServer(MoneyModel, [canvas_element, g_element, gini_chart], "Boltzmann money", model_params=model_params)
    server.launch()
