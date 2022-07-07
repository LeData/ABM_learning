from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.UserParam import UserSettableParameter

from examples.mating import MatingModel

colors = {
    "M": "#ffcc00ff", # yellow
    "F": "#7c00dfff" # purple
}

def agent_shape(agent):
    portrayal = {
                "Layer": 1,
                "Color": "red"
            }
    if agent:
        portrayal["Filled"] = agent.paired,
        portrayal["Color"] = colors[agent.sex]
        if agent.sex == "M":
            portrayal["Shape"] = "circle"
            portrayal["r"] = 1
        else:
            portrayal["Shape"] = "rect"
            portrayal["w"] = .7
            portrayal["h"] = .7
        if agent.paired:
            portrayal["Color"] = "#44444420"
            portrayal["Layer"] = 0
        elif agent.is_incel:
            portrayal["Color"] = "grey"

    return portrayal


if __name__ == "__main__":
    width, height = (20, 10)
    canvas_element = CanvasGrid(agent_shape, grid_height=height, grid_width=width, canvas_width=500, canvas_height=500)
    incel_chart = ChartModule([{"Label": f"Incel_{a_t}", "Color": c} for a_t, c in colors.items()])
    single_chart = ChartModule([{"Label": "Single", "Color": "black"},
                                *[{"Label": f"Single_{a_t}", "Color": c} for a_t, c in colors.items()],
                                ])
    rejections_chart = ChartModule([{"Label": "Avg_rej", "Color": "black"},
                                    *[{"Label": f"Avg_rej_{a_t}", "Color": c} for a_t, c in colors.items()],
                                    ])

    model_params = {
        "height": height,
        "width": width,
        "it_M": UserSettableParameter("slider", "interest_threshold_M", 0.8, 0, 3, step=0.1),
        "it_F": UserSettableParameter("slider", "interest_threshold_F", 2, 0, 3, step=0.1),
        "num_agents": UserSettableParameter("slider", "agent count", 60, 0, 100, step=1),
        "balance": UserSettableParameter("slider", "balance", .5, 0, 1, step=0.05),
    }
    server = ModularServer(MatingModel, [canvas_element, incel_chart, single_chart, rejections_chart], "Mating",
                           model_params=model_params)
    server.launch()
