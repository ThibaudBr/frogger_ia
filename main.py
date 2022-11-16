import math
import os
import pickle
import random

import numpy as np
import pygame
import matplotlib
import matplotlib.pyplot as plt

# Données environnement
FILE_QTABLE = 'qtable.dat'
MAX_STEPS = 200
COOLING_RATE = 0.99
COOLING_RATE_DECAY = 0.9995
ALPHA = 0.1  # Learning Rate
GAMMA = 1
MAP_HEIGHT = 10
MAP_WIDTH = 10
MAX_SESSION = 100000
EXPLORATION = 1
# Reward
DEATH_REWARD = -100
WIN_REWARD = 200
STEP_FORWARD_REWARD = 2
STEP_TO_SIDE_REWARD = -1
STEP_TO_BACK_REWARD = -2
STAY_REWARD = -1
# Positionnement Target et frog
FROG_START_X = 240
FROG_START_Y = 540
FROG_START_MAP_X = int(FROG_START_X / 60)
FROG_START_MAP_Y = int(FROG_START_Y / 60)
TARGET_X = 240
TARGET_Y = 0

FROG_SPEED = 6
# taille des rectangles
FROG_WIDTH = 60
FROG_HEIGHT = 60
TARGET_WIDTH = 60
TARGET_HEIGHT = 60
CAR_WIDTH = 120
CAR_HEIGHT = 60

BASE_HEIGHT = 60
# Liste des actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_STAY = 4
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_STAY]

# Images
FROG_IMAGE = pygame.image.load(os.path.join('textures', 'frog.png'))
CAR_IMAGE = pygame.image.load(os.path.join('textures', 'car.png'))
TARGET_IMAGE = pygame.image.load(os.path.join('textures', 'target.png'))
# plot
PLT_TRONC = 100


class Rectangle:

    def __init__(self, image, rect):
        self.__image = image
        self.__rect = rect

    @property
    def image(self):
        return self.__image

    @property
    def rect(self):
        return self.__rect


class Frog(Rectangle):
    def __init__(self, image, rect, speed):
        super(Frog, self).__init__(image, rect)
        self.__temp_pos_x = self.rect.x
        self.__temp_pos_y = self.rect.y
        self.__speed = speed
        self.__is_move_validated = False
        self.__have_jump = False
        self.__current_pos = (FROG_START_MAP_X, FROG_START_MAP_Y)
        self.__action = None

    def move(self, action):

        if self.__have_jump:
            self.__is_move_validated = False
            self.__temp_pos_x = self.rect.x
            self.__temp_pos_y = self.rect.y

        if action == ACTION_UP:
            if self.rect.y > self.__temp_pos_y - self.rect.h:
                self.rect.y -= self.__speed
                self.__have_jump = False
                if self.rect.y < 0:
                    self.rect.y = 0
                    self.__have_jump = True
            else:
                self.__have_jump = True
                if not self.__is_move_validated:
                    self.__current_pos = (self.__current_pos[0], self.__current_pos[1] - 1)
                    self.__is_move_validated = True
        elif action == ACTION_RIGHT:
            if self.rect.x < self.__temp_pos_x + self.rect.w:
                self.rect.x += self.__speed
                self.__have_jump = False
                if self.rect.x > 540:
                    self.rect.x = 540
                    self.__have_jump = True
            else:
                self.__have_jump = True
                if not self.__is_move_validated:
                    self.__current_pos = (self.__current_pos[0] + 1, self.__current_pos[1])
                    self.__is_move_validated = True

        elif action == ACTION_LEFT:
            if self.rect.x > self.__temp_pos_x - self.rect.w:
                self.rect.x -= self.__speed
                self.__have_jump = False
                if self.rect.x < 0:
                    self.rect.x = 0
                    self.__have_jump = True
            else:
                self.__have_jump = True
                if not self.__is_move_validated:
                    self.__current_pos = (self.__current_pos[0] - 1, self.__current_pos[1])
                    self.__is_move_validated = True

        elif action == ACTION_DOWN:
            if self.rect.y < self.__temp_pos_y + self.rect.h:
                if self.rect.y > 540:
                    self.rect.y = 540
                    self.__have_jump = True
            else:
                self.__have_jump = True
                if not self.__is_move_validated:
                    self.__current_pos = (self.__current_pos[0], self.__current_pos[1] + 1)
                    self.__is_move_validated = True

        elif action == ACTION_STAY:
            self.__have_jump = True
            if not self.__is_move_validated:
                self.__is_move_validated = True

    @property
    def have_jump(self):
        return self.__have_jump

    @property
    def current_pos(self):
        return self.__current_pos

    @have_jump.setter
    def have_jump(self, value):
        self.__have_jump = value

    @current_pos.setter
    def current_pos(self, value):
        self.__current_pos = value


class Car(Rectangle):
    def __init__(self, image, rect):
        super(Car, self).__init__(image, rect)


class Target(Rectangle):
    def __init__(self, image, rect):
        super(Target, self).__init__(image, rect)


class Environment:

    def __init__(self, max_steps=MAX_STEPS, is_q_table=True, cooling_rate=COOLING_RATE, alpha=ALPHA, gamma=GAMMA, max_session=MAX_SESSION, exploration=EXPLORATION):
        self.__windows = pygame.display.set_mode((600, 600))
        self.__frog = None
        self.__target = None
        self.__max_next_q = None
        self.__list_car = []
        self.__collision_map = {(0, 0): 0}
        self.__max_steps = max_steps
        self.__q_table = None
        self.update_q_table(is_q_table)
        self.__reward = 0
        self.__current_state = None
        self.__next_state = None
        self.__cooling_rate = cooling_rate
        self.__max_q_arg = None
        self.__action = None
        self.__exploration = 1
        self.__current_state = None
        self.__alpha = alpha
        self.__gamma = gamma
        self.__session_reward = 0
        self.__max_session = max_session
        self.__list_session_reward = []
        self.__my_font = None
        self.__is_able_to_die = True
        self.init()

    def init(self):
        self.init_frog()
        self.init_target()
        self.init_list_car()
        self.init_font()
        self.init_qtable()
        self.update_windows(0, 0)
        self.init_collision_map()
        self.run_session(True)

    def init_font(self):
        pygame.font.init()
        self.__my_font = pygame.font.SysFont('Comic Sans MS', 30)

    def init_frog(self):
        self.__frog = Frog(FROG_IMAGE, pygame.Rect(FROG_START_X, FROG_START_Y, FROG_WIDTH, FROG_HEIGHT), FROG_SPEED)

    def init_target(self):
        self.__target = Target(TARGET_IMAGE, pygame.Rect(TARGET_X, TARGET_Y, TARGET_WIDTH, TARGET_HEIGHT))

    def init_list_car(self):
        for i in range(8):
            self.__list_car.append(
                Car(CAR_IMAGE, pygame.Rect(random.uniform(0, 540), BASE_HEIGHT * i + 60, CAR_WIDTH, CAR_HEIGHT)))

    def update_windows(self, session, reward):
        self.__windows.fill((0, 0, 0))
        self.__windows.blit(self.__frog.image, (self.__frog.rect.x, self.__frog.rect.y))
        self.__windows.blit(self.__target.image, (self.__target.rect.x, self.__target.rect.y))
        for i in range(len(self.__list_car)):
            if i % 2:
                self.__windows.blit(self.__list_car[i].image, (self.__list_car[i].rect.x, self.__list_car[i].rect.y))
            else:
                self.__windows.blit(pygame.transform.rotate(self.__list_car[i].image, 180),
                                    (self.__list_car[i].rect.x, self.__list_car[i].rect.y))

        text_surface = self.__my_font.render(f"session : {session}, reward : {reward}", False, (255, 255, 255))
        self.__windows.blit(text_surface, (0, 0))
        pygame.display.update()

    def init_collision_map(self):
        for j in range(10):
            for i in range(10):
                self.__collision_map[(i, j)] = 0

    def move_cars(self):
        for i in range(len(self.__list_car)):
            if i % 2:
                self.__list_car[i].rect.x += 3
                if self.__list_car[i].rect.x > 600:
                    self.__list_car[i].rect.x = -self.__list_car[i].rect.w
            else:
                self.__list_car[i].rect.x -= 3
                if self.__list_car[i].rect.x < -self.__list_car[i].rect.w:
                    self.__list_car[i].rect.x = 600

    def update_car_collision(self):
        for car in self.__list_car:
            if car.rect.x >= 0 and car.rect.w <= 600:
                pos_x = car.rect.x // 60
                pos_y = car.rect.y // 60
                car_hit_box = CAR_WIDTH // 60
                self.update_collision_map(pos_x, pos_y, car_hit_box)

    def update_collision_map(self, pos_x, pos_y, car_hit_box):
        if pos_y == 2:
            pos_x -= 1

        for i in range(10):
            self.__collision_map[(i, pos_y)] = 0

        for i in range(10):
            if i >= pos_x and i <= pos_x + car_hit_box:
                self.__collision_map[(i, pos_y)] = 1

    def check_collision(self):
        for car in self.__list_car:
            if self.__frog.rect.colliderect(car.rect):
                return True, False
        if self.__frog.rect.colliderect(self.__target.rect):
            return False, True
        return False, False

    def update_q_table(self, is_q_table):
        if is_q_table:
            if os.path.exists(FILE_QTABLE):
                old_table = open(FILE_QTABLE, 'r')
                self.__q_table = pickle.load(old_table)
            else:
                pass
        else:
            pass

    def init_qtable(self):
        self.__q_table = {}
        # for state in range(MAP_HEIGHT * MAP_WIDTH):
        #     self.__q_table[state] = {}
        #     for action in ACTIONS:
        #         self.__q_table[state][action] = 0
        for up in range(5):
            for right in range(5):
                for left in range(5):
                    for down in range(5):
                        for stay in range(5):
                            for current in range(10):
                                self.__q_table[up, right, left, down, stay, current] = [math.floor(random.uniform(0,6)) for i in range(5)]

    def save(self):
        with open(FILE_QTABLE, 'wb') as file:
            pickle.dump(self.__q_table, file)

    def update_state(self, is_next):

        state_stay = self.__collision_map[(self.__frog.current_pos[0], self.__frog.current_pos[1])]

        if self.__frog.current_pos[1] > 0:
            state_up = self.__collision_map[(self.__frog.current_pos[0], self.__frog.current_pos[1] - 1)]
        else:
            state_up = 2

        if self.__frog.current_pos[1] < 9:
            state_down = self.__collision_map[(self.__frog.current_pos[0], self.__frog.current_pos[1] + 1)]
        else:
            state_down = 2

        if self.__frog.current_pos[0] < 9:
            state_right = self.__collision_map[(self.__frog.current_pos[0] + 1, self.__frog.current_pos[1])]
        else:
            state_right = 2

        if self.__frog.current_pos[0] > 0:
            state_left = self.__collision_map[(self.__frog.current_pos[0] - 1, self.__frog.current_pos[1])]
        else:
            state_left = 2

        if not is_next:
            self.__current_state = (state_up, state_down, state_right, state_left, state_stay, self.__frog.current_pos[1])
            if np.random.uniform(0,1) < self.__exploration:
                self.__exploration *= self.__cooling_rate
                self.__max_q_arg = np.argmax(self.__q_table[self.__current_state])
                self.__action = self.__max_q_arg + 1
            else:
                self.__action = np.random.randint(1, 6)

            if self.__action == ACTION_UP and state_up == 2:
                return True
            if self.__action == ACTION_RIGHT and state_right == 2:
                return True
            if self.__action == ACTION_LEFT and state_left == 2:
                return True
            if self.__action == ACTION_DOWN and state_down == 2:
                return True

            return False
        else:
            self.__next_state = (state_up, state_down, state_right, state_left, state_stay, self.__frog.current_pos[1])
            return False

    def run_game(self, session):
        is_loose_game = False
        is_won_game = False
        step = 0
        self.__action = ACTION_STAY
        is_out_of_map = False
        new_q = None

        while step < MAX_STEPS:
            if is_out_of_map:
                self.__frog.have_jump = True

            self.move_cars()
            self.update_car_collision()

            if self.__frog.have_jump:
                step += 1

                is_out_of_map = self.update_state(False)

                if not is_out_of_map:
                    self.__frog.move(self.__action)
                    is_loose_game, is_won_game = self.check_collision()

                if is_loose_game or is_out_of_map:
                    self.__reward = DEATH_REWARD
                elif is_won_game:
                    self.__reward = WIN_REWARD
                else:
                    if self.__action == ACTION_LEFT or self.__action == ACTION_RIGHT:
                        self.__reward = STEP_TO_SIDE_REWARD
                    elif self.__action == 1:
                        self.__reward = STEP_FORWARD_REWARD
                    elif self.__action == 4:
                        self.__reward = STEP_TO_BACK_REWARD
                    else:
                        self.__reward = STAY_REWARD

                if is_loose_game:
                    new_q = DEATH_REWARD
                elif is_won_game:
                    new_q = WIN_REWARD
                else:
                    is_out_of_map = self.update_state(True)

                    self.__max_next_q = max(self.__q_table[self.__next_state])
                    # new_q = self.__alpha * (
                    #             self.__reward + self.__gamma * self.__max_next_q - self.__q_table[self.__current_state][self.__action])

                    new_q = (1 - self.__alpha) * self.__q_table[self.__current_state][self.__action - 1] + self.__alpha * (self.__reward + self.__max_next_q)
                self.__q_table[self.__current_state][self.__action - 1] = new_q

                if (self.__reward == WIN_REWARD or self.__reward == DEATH_REWARD or step == self.__max_steps) and self.__is_able_to_die:
                    self.__frog.rect.x = 240
                    self.__frog.rect.y = 540
                    self.__frog.current_pos = (4, 9)
                    self.__session_reward += self.__reward
                    break
            else:
                self.__frog.move(self.__action)
                self.check_collision()

            self.__session_reward += self.__reward
            self.update_windows(session, self.__reward)
        print(f"episode: {session}, cooling_rate: {self.__cooling_rate}, reward: {self.__session_reward}, exploration: {self.__exploration}")
        self.__list_session_reward.append(self.__session_reward)
        self.__session_reward = 0
        self.__cooling_rate *= COOLING_RATE_DECAY

    def run_session(self, is_should_save= True):
        for session in range(self.__max_session):
            self.run_game(session)

        rewardsAverage = np.convolve(self.__list_session_reward, np.ones((PLT_TRONC,)) / PLT_TRONC, mode="valid")
        plt.plot([i for i in range(len(rewardsAverage))], rewardsAverage)
        plt.ylabel(f"reward {PLT_TRONC}ma")
        plt.xlabel("Session #")
        plt.show()

        if is_should_save:
            self.save()


if __name__ == "__main__":
    env = Environment()
