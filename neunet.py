import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys
import os
import time
import cv2
import random

from torch import Tensor

from main_for_ai import Game

from main import Game as Play_Game
import pygame as pg


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.number_of_actions = 2
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 2000000
        self.replay_memory_size = 10000
        self.minibatch_size = 32

        self.conv1 = nn.Conv2d(4, 32, 5, 4)
        self.conv2 = nn.Conv2d(32, 64, 5, 2)
        self.conv3 = nn.Conv2d(64, 128, 5, 1)

        self.fc4 = nn.Linear(2048, 512)
        self.fc5 = nn.Linear(512, self.number_of_actions)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.relu4 = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.conv1(x)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.conv3(output)
        output = self.relu3(output)

        output = output.view(output.size()[0], -1)
        # print("before fc4: ", output.size())
        output = self.fc4(output)
        output = self.relu4(output)
        # print("after fc4: ", output.size())
        output = self.fc5(output)

        return output


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)


def resize_and_bgr2gray(img):
    # img = img[0:600, 0:600]
    image_data = cv2.cvtColor(cv2.resize(img, (84, 84)), cv2.COLOR_BGR2GRAY)    # to grayscale
    image_data[image_data > 0] = 255    # ako nije crno (0), tj sivo je, onda pretvori u bijelo (255); nema sivih tonova
    image_data = np.reshape(image_data, (84, 84, 1))    # resize

    return image_data


def image_to_tensor(img):
    img_tensor = img.transpose(2, 0, 1)             # transponiranje matrice; x,y,z --> z,x,y
    img_tensor = img_tensor.astype(np.float32)      # pretvaranje podataka u float
    img_tensor = torch.from_numpy(img_tensor)       # pretvaranje u torch tensor

    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()              # ako je cuda available tensor ide na cuda

    return img_tensor                               # vraća tensor


def train(model, start):
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    criterion = nn.MSELoss()

    game_state = Game()

    replay_memory = []


    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1

    # image_data = game_state.new(action)[0]
    # reward = game_state.new(action)[1]
    # terminal = game_state.new(action)[2]

    image_data, reward, terminal = game_state.new(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)

    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    epsilon = model.initial_epsilon
    iteration = 0

    epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)

    while iteration < model.number_of_iterations:       # main loop
        output = model(state)[0]                        # dobi output iz nn
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)

        if torch.cuda.is_available():
            action = action.cuda()

        random_action = random.random() <= epsilon

        if random_action:
            print("Performed random action!")
        action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0]

        if torch.cuda.is_available():
            action_index = action_index.cuda()

        action[action_index] = 1

        image_data_1, reward, terminal = game_state.new(action)
        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)

        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)
        action = action.unsqueeze(0)
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

        replay_memory.append((state, action, reward, state_1, terminal))

        if len(replay_memory) > model.replay_memory_size:
            replay_memory.pop(0)

        epsilon = epsilon_decrements[iteration]

        minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))

        state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state_1_batch = state_1_batch.cuda()

        output_1_batch = model(state_1_batch)

        y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                  else reward_batch[i] + model.gamma * torch.max(output_1_batch[i])
                                  for i in range(len(minibatch))))

        q_value = torch.sum(model(state_batch) * action_batch, dim=1)

        optimizer.zero_grad()

        y_batch = y_batch.detach()

        loss = criterion(q_value, y_batch)
        loss.backward()
        optimizer.step()

        state = state_1
        iteration += 1

        if iteration % 200 == 0:
            torch.save(model, "pretrained_model_2/current_model_" + str(iteration) + ".pth")

        print("iteration: ", iteration, "elapsed time: ", time.time() - start, "epsilon: ", epsilon,
              "action: ", action_index.cpu().detach().numpy(), "reward: ", reward.numpy()[0][0],
              "Q max: ", np.max(output.cpu().detach().numpy()))


def test(model):
    game_state = Game()

    # input_actions[0] == 1: stoji
    # input_actions[1] == 1: skači

    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1

    image_data, reward, terminal = game_state.new(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)

    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    while True:
        output = model(state)[0]

        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():
            action = action.cuda()

        action_index = torch.argmax(output)
        if torch.cuda.is_available():
            action_index = action_index.cuda()
        action[action_index] = 1

        image_data_1, reward, terminal = game_state.new(action)
        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        state = state_1


def main(mode):
    # Provjera ako je cuda available
    cuda_is_available = torch.cuda.is_available()

    if mode == 'test':
        model = torch.load(
            'pretrained_model_2/current_model_9200.pth',
            map_location='cpu' if not cuda_is_available else None
        ).eval()

        if cuda_is_available:
            model = model.cuda()

        test(model)

    elif mode == 'train':
        if not os.path.exists('pretrained_model_2/'):     # ako ne postoji folder
            os.mkdir('pretrained_model_2/')               # napravi ga

        model = Net()        #  za novi model

        # za učitati postojeći model
        """
        model = torch.load(
            'pretrained_model/current_model_3600.pth',
            map_location='cpu' if not cuda_is_available else None
        ).eval()
        """

        if cuda_is_available:       # ako je cuda available
            model = model.cuda()    # model ide na GPU

        model.apply(init_weights)   # ...
        start = time.time()         # početno vrijeme

        train(model, start)         # započinje treniranje

    elif mode == 'play':
        game = Play_Game()

        while game.running:
            game.new()

        pg.quit()


if __name__ == "__main__":
    main(sys.argv[1])

