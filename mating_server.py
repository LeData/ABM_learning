from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.UserParam import UserSettableParameter

from examples.mating import MatingModel


def agent_shape(agent):
    portrayal = {
                "Shape": "circle",
                "r": 1,
                "Layer": 0,
                "Color": "black"
            }
    if agent:
        portrayal["Filled"] = agent.paired,
        portrayal["r"] = 1 - .5 * int(agent.paired)
        portrayal["Color"] = "blue" if agent.type == "male" else "pink"
        if agent.is_incel:
            portrayal["Color"] = "grey"

    return portrayal


if __name__ == "__main__":
    width, height = (20, 10)
    canvas_element = CanvasGrid(agent_shape, grid_height=height, grid_width=width, canvas_width=500, canvas_height=500)
    incel_chart = ChartModule([{"Label": "Incel_prop", "Color": "Black"}])

    model_params = {
        "height": height,
        "width": width,
        "num_agents": UserSettableParameter("slider", "agent count", 20, 0, 100, step=1),
        "balance": .6
    }
    server = ModularServer(MatingModel, [canvas_element, incel_chart], "Mating", model_params=model_params)
    server.launch()