import matplotlib.pyplot as plt
import pandas as pd


def run(results_directory, optimizer, objectivefunc, Iterations):
    plt.ioff()
    fileResultsData = pd.read_csv(results_directory + "/experiment.csv")

    for j in range(0, len(objectivefunc)):
        objective_name = objectivefunc[j]

        startIteration = 0
        if "SSA" in optimizer:
            startIteration = 1
        allGenerations = [x + 1 for x in range(startIteration, Iterations)]
        for i in range(len(optimizer)):
            optimizer_name = optimizer[i]

            row = fileResultsData[
                (fileResultsData["Optimizer"] == optimizer_name)
                & (fileResultsData["objfname"] == objective_name)
            ]
            row = row.iloc[:, 3 + startIteration :]
            plt.plot(allGenerations, row.values.tolist()[0], label=optimizer_name)
        plt.xlabel("Iterations")
        plt.ylabel("Fitness")
        plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.02))
        plt.grid()
        fig_name = results_directory + "/convergence-" + objective_name + ".png"
        plt.savefig(fig_name, bbox_inches="tight")
        plt.clf()
        # plt.show()
