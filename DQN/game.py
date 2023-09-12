import sys
from typing import Tuple, Union
import pygame
import random
import numpy as np
import cv2


class FlappyBird: 
    def __init__(self, CNN_mode: bool = False,
                res: Tuple[int, int] = (432, 576)) -> None:
        """
        Sets resoultion and initalizes a game instance.
        

        Args:
            CNN_res (Tuple[int, int]): Resolution of returned frame in
            CNN mode, None if xy state mode.
            res (Tuple[int, int], optional): Resolution in xy mode. 
            Scaled down by self.res_ratio in CNN mode for better
            performance. Defaults to (432, 576).
        """

        self.CNN_mode = CNN_mode
        self.res = res

        # Display resolution scaler, 1 if not in CNN_mode.
        self.res_ratio = 1

        if CNN_mode:
            """
            Can't be less than 1/3 for default resolution for proper
            displaying.
            """
            self.res_ratio = (
                1 / 3
            )

            # Scaling resoultion in CNN mode
            self.res = (
                self.res[0] * self.res_ratio,
                self.res[1] * self.res_ratio,
            )

        # Additional resize for returned frames
        self.image_resize = 0.5    

        # Initialize pygame engine
        pygame.init()

        # Set resolution
        self.screen = pygame.display.set_mode(self.res)
       
        # Flag for key state
        self.pressed = 0

        # Current animation frame
        self.animation = 0

        # Flag for terminated episode
        self.done = 0

        # Load background texture
        self.background_text = pygame.image.load("textures/bg.png")

        # Scale to match resolution
        self.background_text = pygame.transform.scale(
            self.background_text, self.res
        )
        self.background_rect = self.background_text.get_rect()

        # Load ground texture
        self.ground_text = pygame.image.load("textures/ground.png")

        # Scale to match resolution
        self.ground_text = pygame.transform.scale(
            self.ground_text, (577 * self.res_ratio, 112 * self.res_ratio)
        )
        self.ground_rect = self.ground_text.get_rect()

        """
        Ground consists of two textures side by side, when texture goes
        completely off screen it moves back to the right
        """
        self.ground_rect_pair = [
            self.ground_text.get_rect(),
            self.ground_text.get_rect(),
        ]

        # Set position for both ground textures
        self.ground_rect_pair[0].x = 0
        self.ground_rect_pair[0].y = self.ground_rect_pair[1].y = (
            500 * self.res_ratio)
        self.ground_rect_pair[1].x = 577 * self.res_ratio  # x
        
        # Initial reward
        self.reward = 0

        # Create bird object
        self.bird = Bird(self.res, self.res_ratio)

        # Create two pipe objects
        self.pipes = []

        self.pipes.append(
            Pipes(600 * self.res_ratio, self.res, self.res_ratio))

        self.pipes.append(
            Pipes(
                (
                    600 * self.res_ratio
                    + (self.res[0] - 90 * self.res_ratio) / 2
                    + 90 * self.res_ratio),
                self.res,
                self.res_ratio)
        )

        # Rectangles for displaying x and y position of the bird
        self.state_rects = []
        self.state_rects.extend(
            (pygame.Rect(10, 430, 100, 10), pygame.Rect(10, 460, 100, 10))
        )

        # Rectangles for displaying Q values for both decisions
        self.action_rects = []
        self.action_rects.extend(
            (
                pygame.Rect(
                    self.bird.rect.x + self.bird.rect.width / 8, 0, 10, 0
                ),
                pygame.Rect(
                    self.bird.rect.x + self.bird.rect.width / 8, 0, 10, 0
                ),
            )
        )

        # Index of pipe closest the player
        self.closest_pipe_index = 0


    def animate_ground(self) -> None:
        """Move both ground textures
        """
        self.ground_rect_pair[0].x -= 10 * self.res_ratio
        self.ground_rect_pair[1].x -= 10 * self.res_ratio

        # Move the the back when off screen
        if self.ground_rect_pair[0].x <= -577 * self.res_ratio:
            self.ground_rect_pair[0].x = 576 * self.res_ratio
        if self.ground_rect_pair[1].x <= -577 * self.res_ratio:
            self.ground_rect_pair[1].x = 576 * self.res_ratio

    def reset(self) -> Union[Tuple[float, float], Tuple[np.ndarray]]:
        """Resets the envoirnment and returs initial state

        Returns:
            Union[Tuple[float, float], Tuple[np.ndarray]]
        """

        # Reset bird  
        self.bird.reset()

        # Reset pipes
        self.pipes[0].reset()
        self.pipes[1].reset()
        self.closest_pipe_index = 0

        # Reset reward
        self.reward = 0

        self.done = 0
        return self.get_state()

    def render(self, q_0: float, q_1: float) -> None:
        """Render next frame

        Renders 

        Args:
            q_0 (float): Q value for action 0. Displayed with rectangle.
            q_1 (float): Q value for action 1. Displayed with rectangle.
        """        

        # Events handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        self.animate_ground()  #needs fixing
        self.screen.blit(self.background_text, self.background_rect)
        self.screen.blit(
            self.bird.texture,
            self.bird.rect,
            area=(
                (int)(self.animation) % 3 * 51 * self.res_ratio,
                0,
                51 * self.res_ratio,
                36 * self.res_ratio,
            ),
        )
        self.screen.blit(self.ground_text, self.ground_rect_pair[0])
        self.screen.blit(self.ground_text, self.ground_rect_pair[1])

        for p in self.pipes:
            self.screen.blit(
                p.txt[0],
                p.rect[0],
                area=(
                    0,
                    0,
                    p.rect[0].width,
                    self.ground_rect_pair[0].y - p.height,
                ),
            )
            self.screen.blit(p.txt[1], p.rect[1])

        self.animation += 0.3

        if not self.CNN_mode:
            for s in self.state_rects:
                states = self.get_state()
                s.width = states[self.state_rects.index(s)] * 100
                pygame.draw.rect(self.screen, (50, 50, 255), s)

            # predictions
            sub = 2000 * (q_0 - q_1)
            if sub >= 0:
                q_0 = sub
                q_1 = 0
            if sub < 0:
                q_1 = sub * -1
                q_0 = 0
            self.action_rects[0].height = q_0
            self.action_rects[0].y = (
                self.bird.rect.y + self.bird.rect.height + 10
            )
            pygame.draw.rect(self.screen, (255, 50, 50), self.action_rects[0])
            self.action_rects[1].height = q_1
            self.action_rects[1].y = self.bird.rect.y - q_1 - 10
            pygame.draw.rect(self.screen, (255, 50, 50), self.action_rects[1])

        pygame.display.update()

    def step(self, action: int) -> Tuple[any, int, int, str]:
        self.bird.move(action)
        for p in self.pipes:
            p.move()

        if self.bird.check_collision(
            self.pipes[0].rect[0],
            self.pipes[0].rect[1],
            self.pipes[1].rect[0],
            self.pipes[1].rect[1],
        ):
            self.done = 1

        # Reward for surviving
        self.reward = 0

        # Get current state. May change reward.
        state = self.get_state()

        # Additional informations
        info = ""

        return (state, self.reward, self.done, info)

    def get_state(self) -> Union[Tuple[float, float], Tuple[np.ndarray]]:
        """Returns current state.

        Depending on the mode the state can be x and y distance from
        the closest pipe or current frame of the game.

        Returns:
            Union[Tuple[float, float], Tuple[np.ndarray]]
        """        

        # X distance to closest pipe
        dist_x = (
            self.pipes[self.closest_pipe_index].rect[0].x
            + self.pipes[self.closest_pipe_index].rect[0].width
            - self.bird.rect.x
        )

        # Y distance the closest pipe
        dist_y = self.pipes[self.closest_pipe_index].rect[0].y - (
            self.bird.rect.y + self.bird.rect.height
        )

        # Bird goes through the pipe
        if dist_x < 0:
            # Reward for passing the obstacle
            self.reward = 1
            if not self.closest_pipe_index:
                self.closest_pipe_index = 1
            else:
                self.closest_pipe_index = 0
        else:
            self.reward = 0

        # CNN mode
        if self.CNN_mode:
            image = pygame.surfarray.array3d(self.screen)
            image = np.swapaxes(image, 0, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(
                image,
                (
                    int(image.shape[1] * self.image_resize),
                    int(image.shape[0] * self.image_resize),
                ),
                interpolation=cv2.INTER_AREA,
            )  # resize
            image = image / 255  # normalization
            return image

        return (dist_x / 522.0, dist_y / 600.0 + 0.5)


class Bird:
    def __init__(self, res, res_ratio):
        self.res_ratio = res_ratio
        self.texture = pygame.image.load("textures/flappybird.png")
        self.texture = pygame.transform.scale(
            self.texture, (153 * self.res_ratio, 36 * self.res_ratio)
        )
        self.rect = self.texture.get_rect()
        self.rect.x = res[0] / 2 - self.rect.width / 3
        self.frame = -1
        self.rotation = 0

    def move(self, action):
        if action == 1 and self.frame < 4:
            self.frame = 6

        if self.frame > 0:
            self.rect.y -= (int)(
                self.frame * self.frame * 0.6 * self.res_ratio
            )
            self.frame -= 1

        elif self.frame == 0:
            self.frame -= 1
        else:
            self.rect.y += 19 * self.res_ratio

    def check_collision(self, p1, p2, p3, p4):
        self.hitbox = pygame.Rect(
            self.rect.x, self.rect.y, self.res_ratio * 51, self.res_ratio * 36
        )
        cl = lambda p: self.hitbox.colliderect(p)

        if (
            cl(p1)
            or cl(p2)
            or cl(p3)
            or cl(p4)
            or self.rect.y + self.rect.height >= self.res_ratio * 500
            or self.rect.y <= -self.res_ratio * 50
        ):
            return True

        else:
            return False

    def reset(self):
        self.rect.y = self.res_ratio * 200
        self.frame = -1


class Pipes:
    def __init__(self, position, res, res_ratio):
        self.res_ratio = res_ratio
        self.position = position
        self.scale = (90 * res_ratio, 313 * self.res_ratio)
        self.txt_d = pygame.image.load("textures/pipe.png")
        self.txt_u = pygame.image.load("textures/piper.png")
        self.txt_d = pygame.transform.scale(self.txt_d, self.scale)
        self.txt_u = pygame.transform.scale(self.txt_u, self.scale)
        self.txt = []
        self.txt.append(self.txt_d)
        self.txt.append(self.txt_u)
        self.rect = []
        self.rect.extend([self.txt_d.get_rect(), self.txt_u.get_rect()])
        self.upper_lim = int(210 * self.res_ratio)
        self.bottom_lim = int(440 * self.res_ratio)
        self.height = random.randint(self.upper_lim, self.bottom_lim)
        self.gap = 140 * self.res_ratio
        self.rect[0].x = self.rect[1].x = position
        self.rect[0].y, self.rect[1].y = (
            self.height,
            self.height - self.rect[1].height - self.gap,
        )
        self.res = res

    def move(self):
        self.rect[0].x -= 10 * self.res_ratio
        self.rect[1].x -= 10 * self.res_ratio
        if self.rect[0].x < -self.rect[0].width:
            self.rect[0].x = self.res[0]
            self.rect[1].x = self.res[0]
            self.height = random.randint(self.upper_lim, self.bottom_lim)
            self.rect[0].y, self.rect[1].y = (
                self.height,
                self.height - self.rect[1].height - self.gap,
            )

    def reset(self):
        self.rect[0].x = self.rect[1].x = self.position
        self.height = random.randint(self.upper_lim, self.bottom_lim)
        self.rect[0].y, self.rect[1].y = (
            self.height,
            self.height - self.rect[1].height - self.gap,
        )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--tps', type=int, default=25, help='Ticks per second')
    args = parser.parse_args()
    tps = args.tps

    env = FlappyBird()

    clock = pygame.time.Clock()
    running = True
    key = pygame.K_UP
    key_state = {}

    while running:
        env.reset()
        score = 0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            keys = pygame.key.get_pressed()

            action = 0
            if pygame.key.get_pressed()[key] and not key_state.get(key, False):
                key_state[key] = True
                action = 1

            elif not pygame.key.get_pressed()[key]:
                key_state[key] = False
            
            _, reward, done, _ = env.step(action)
            score += reward
            
            env.render(0, 0)
            clock.tick(tps)

            if done:
                print(f"Score: {score}")
                break