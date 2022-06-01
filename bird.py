import sys
import pygame
import time
import random
import os



class FlappyBird:
    def __init__(self):
        pygame.init()
        self.RES = (288*1.5, 384*1.5)
        self.FPS = 60
        self.screen = pygame.display.set_mode(self.RES)   # resolution

        self.background_text = pygame.image.load('flappy/bg.png')
        self.background_text = pygame.transform.scale(self.background_text, self.RES)
        self.background_rect = self.background_text.get_rect()
        self.ground_text = pygame.image.load('flappy/ground.png')
        self.ground_rect = self.ground_text.get_rect()
        self.ground_rect_pair = [self.ground_text.get_rect(), self.ground_text.get_rect()]
        self.ground_rect_pair[0].x = 577
        self.ground_rect_pair[1].x = 0
        self.ground_rect_pair[0].y = self.ground_rect_pair[1].y = 500

        self.reward = 0
        self.bird = Bird(self.RES)
        self.pipes = []
        self.pipes.append(Pipes(600, self.RES))
        self.pipes.append(Pipes(600+(self.RES[0]-90)/2+90, self.RES))
        self.clock = pygame.time.Clock()
        self.pressed = 0
        self.ticks = 0
        self.animation = 0
        self.done = 0

        self.state_rects = []
        self.state_rects.extend( (pygame.Rect(10, 430, 100, 10), pygame.Rect(10, 460, 100, 10), pygame.Rect(10, 490, 100, 10), pygame.Rect(10, 520, 100, 10)) )
        self.action_rects = []
        self.action_rects.extend( (pygame.Rect(self.bird.rect.x+self.bird.rect.width/8, 0, 10, 0), pygame.Rect(self.bird.rect.x+self.bird.rect.width/8, 0, 10, 0)) )


        self.closest_pipe_index = 0

    def animate_ground(self):
        self.ground_rect_pair[0].x += -3
        self.ground_rect_pair[1].x += -3

        if self.ground_rect_pair[0].x < -577: self.ground_rect_pair[0].x = 574
        if self.ground_rect_pair[1].x < -577: self.ground_rect_pair[1].x = 574


    #Resets the environment and returns intial state
    def reset(self):
        self.bird.reset()
        self.pipes[0].reset()
        self.pipes[1].reset()
        self.reward = 0
        self.done = 0
        self.closest_pipe_index = 0
        return self.get_state()


    def render(self, q_0, q_1):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()

        self.animate_ground()
        self.screen.fill((212, 99, 19))
        self.screen.blit(self.background_text, self.background_rect)
        self.screen.blit(self.bird.texture, self.bird.rect, area=((int)(self.animation)%3*51, 0, 51, 36))
        self.screen.blit(self.ground_text, self.ground_rect_pair[0])
        self.screen.blit(self.ground_text, self.ground_rect_pair[1])

        for p in self.pipes:
            self.screen.blit(p.txt[0], p.rect[0], area=(0,0, p.rect[0].width, self.ground_rect_pair[0].y-p.height))
            self.screen.blit(p.txt[1], p.rect[1])

        self.animation += 0.3

        for s in self.state_rects:
            states = self.get_state()
            states[0] = states[0]/3
            states[1] = states[1]*7
            states[2] = states[2]/3
            states[3] = states[3]/3
            s.width = states[self.state_rects.index(s)]
            pygame.draw.rect(self.screen, (50, 50, 255), s)

        q_0 = q_0
        q_1 = q_1
        sub = q_0 - q_1
        sub = sub/2.5
        if sub >= 0:
            q_0 = sub
            q_1 = 0
        if sub < 0:
            q1 = sub * -1
            q0 = 0
        self.action_rects[0].height = q_0
        self.action_rects[0].y = self.bird.rect.y + self.bird.rect.height + 10
        pygame.draw.rect(self.screen, (255,50,50), self.action_rects[0])
        self.action_rects[1].height = q_1
        self.action_rects[1].y = self.bird.rect.y - q_1 - 10
        pygame.draw.rect(self.screen, (255,50,50), self.action_rects[1])


        pygame.display.update()
        self.clock.tick(self.FPS)


    def step(self, action):
        self.bird.move(action)
        for p in self.pipes:
            p.move()
        self.reward += 1
        if self.bird.check_collision(self.pipes[0].rect[0], self.pipes[0].rect[1], self.pipes[1].rect[0], self.pipes[1].rect[1]):
            self.done = 1

        return [self.get_state(), self.reward, self.done, " "]


    def get_state(self):
        dist = self.pipes[self.closest_pipe_index].rect[0].x + self.pipes[self.closest_pipe_index].rect[0].width - (self.bird.rect.x)
        if dist < 0:
            self.closest_pipe_index += 1
            if(self.closest_pipe_index)==2: self.closest_pipe_index = 0

        return [self.bird.rect.y, self.bird.frame, dist, self.pipes[self.closest_pipe_index].rect[0].y]

class Bird:
    def __init__(self, RES):
        self.texture = pygame.image.load('flappy/flappybird.png')
        self.texture = pygame.transform.scale(self.texture, (153, 36))
        self.rect = self.texture.get_rect()
        self.rect.x = RES[0]/2 - self.rect.width/3
        self.frame = -1
        self.rotation = 0

    def move(self, action):
        if action == 1 and self.frame < 10:
            self.frame = 16

        if self.frame > 0:
            self.rect.y -= (int)(self.frame*self.frame*0.05)
            self.frame -= 1

        elif self.frame == 0:
            self.frame -= 1
        else:
            self.rect.y += 8.8


    def check_collision(self, p1, p2, p3, p4):
        self.hitbox = pygame.Rect(self.rect.x, self.rect.y, 51, 36)
        cl = lambda p: self.hitbox.colliderect(p)

        if cl(p1) or cl(p2) or cl(p3) or cl(p4) or self.rect.y + self.rect.height >= 500 or self.rect.y <= -50:
            return True

        else:
            return False


    def reset(self):
        self.rect.y = 200
        self.frame = -1



class Pipes:
    def __init__(self, position, RES):
        self.position = position
        self.scale = (90, 313)
        self.txt_d = pygame.image.load('flappy/pipe.png')
        self.txt_u = pygame.image.load('flappy/piper.png')
        self.txt_d = pygame.transform.scale(self.txt_d, self.scale)
        self.txt_u = pygame.transform.scale(self.txt_u, self.scale)
        self.txt = []
        self.txt.append(self.txt_d)
        self.txt.append(self.txt_u)
        self.rect = []
        self.rect.extend([self.txt_d.get_rect(), self.txt_u.get_rect()])
        self.upper_lim = 210
        self.bottom_lim = 440
        self.height = random.randint(self.upper_lim, self.bottom_lim)
        self.gap = 140
        self.rect[0].x = self.rect[1].x = position
        self.rect[0].y, self.rect[1].y = self.height, self.height-self.rect[1].height-self.gap
        self.RES = RES

    def move(self):
        self.rect[0].x -= 3
        self.rect[1].x -= 3
        if self.rect[0].x < -self.rect[0].width:
            self.rect[0].x = self.RES[0]
            self.rect[1].x = self.RES[0]
            self.height = random.randint(self.upper_lim, self.bottom_lim)
            self.rect[0].y, self.rect[1].y = self.height, self.height-self.rect[1].height-self.gap

    def reset(self):
        self.rect[0].x = self.rect[1].x = self.position
        self.height = random.randint(self.upper_lim, self.bottom_lim)
        self.rect[0].y, self.rect[1].y = self.height, self.height-self.rect[1].height-self.gap
