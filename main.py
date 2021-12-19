from madqn import MADQN
import pickle
from PIL import Image, ImageDraw
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


def gif_maker():
    dimension = 50

    pickle_off = open("results_412050network.pkl", 'rb')
    emp = pickle.load(pickle_off)

    simList = list(emp[0]['sim_states'])

    forestList = []

    for episode in simList:
        # sim.update(control)

        image = np.zeros((dimension, dimension, 3), dtype=np.uint8)

        for i in range(dimension):
            for j in range(dimension):
                if episode[i][j] == 0:
                    image[i][j] = [0, 255, 0]  # Green
                elif episode[i][j] == 1:
                    image[i][j] = [255, 0, 0]  # Red
                elif episode[i][j] == 2:
                    image[i][j] = [87, 87, 87]  # Grey
                else:
                    pass

        displaySim = Image.fromarray(image, 'RGB')
        displaySim = displaySim.resize((800, 800))
        forestList.append(displaySim)

    forestList[0].save('forest331.gif', save_all=True, append_images=forestList[1:], duration=150, loop=0)


def train_network(episodes):
    # train and save a network
    algorithm = MADQN(mode='train')
    algorithm.train(num_episodes=episodes)


def test_network(network_file):
    # test a network or the heuristic
    load_filename = network_file
    test_method = 'network'
    algorithm = MADQN(mode='test', filename=load_filename)
    results = algorithm.test(num_episodes=1, method=test_method)

    # # save the results to file
    # # WARNING: the resulting output file may be very large, ~1.5 GB
    save_filename = 'results_412050' + test_method + '.pkl'
    output = open(save_filename, 'wb')
    pickle.dump(results, output)
    output.close()


def sim_output(results_file):
    dimension = 50

    pickle_off = open(results_file, 'rb')
    emp = pickle.load(pickle_off)

    simList = list(emp[0]['sim_states'])
    team = emp[0]['team']

    forestList = []
    gifList = []

    for a in range(len(simList)):
        # sim.update(control)

        image = np.zeros((dimension, dimension, 3), dtype=np.uint8)  # 50x50 grid

        for i in range(dimension):
            for j in range(dimension):
                if simList[a][i][j] == 0:
                    image[i][j] = [0, 255, 0]  # Green
                elif simList[a][i][j] == 1:
                    image[i][j] = [255, 0, 0]  # Red
                elif simList[a][i][j] == 2:
                    image[i][j] = [87, 87, 87]  # Grey
                else:
                    pass

        row_labels = range(dimension)
        col_labels = range(dimension)
        plt.matshow(image)

        for agent in team:
            if a > 1:
                x_line = []
                y_line = []
                x, y = team[agent].positions[a]
                plt.plot(x, y)
                plt.text(x, y, f'{team[agent].numeric_id}')
                for m in range(2, a):
                    x_line.append(team[agent].positions[m][0])
                    y_line.append(team[agent].positions[m][1])
                plt.plot(x_line, y_line, 'b-')

        # plt.xticks(range(dimension), col_labels)
        # plt.yticks(range(dimension), row_labels)
        plt.show()
    #    plt.pause(.5)


if __name__ == '__main__':
     # algorithm = MADQN(mode='train')
     # algorithm.train(num_episodes=10)
    #test_network('madqn-18-Dec-2021-0945.pth.tar')
    sim_output('results_412050network.pkl')
    #gif_maker()


