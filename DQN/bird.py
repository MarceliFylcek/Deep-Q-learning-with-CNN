import sys
import pygame
import time
import random
import os
import numpy as np
import cv2


class FlappyBird:
    def __init__(self, CNN_res):
        self.CNN_res = CNN_res
        pygame.init()
        self.res = (432, 576) #default resolution
        self.res_ratio = 1
        if CNN_res:
            self.res_ratio = 1/3 #size of rendered screen compared to default
            #affects only rendered resolution, image is then scaled down by image_resize before returning
            #1/2 216 288
            #1/3 144 192 (min)
            self.res = (self.res[0]*self.res_ratio, self.res[1]*self.res_ratio)

        self.FPS = 15
        self.screen = pygame.display.set_mode(self.res)

        self.background_text = pygame.image.load('flappy/bg.png')
        self.background_text = pygame.transform.scale(self.background_text, self.res)
        self.background_rect = self.background_text.get_rect()
        self.ground_text = pygame.image.load('flappy/ground.png')
        self.ground_text = pygame.transform.scale(self.ground_text, (577*self.res_ratio, 112*self.res_ratio))
        self.ground_rect = self.ground_text.get_rect()
        self.ground_rect_pair = [self.ground_text.get_rect(), self.ground_text.get_rect()]
        self.ground_rect_pair[0].x = 0                                                       #x
        self.ground_rect_pair[1].x = 577 * self.res_ratio                                    #x
        self.ground_rect_pair[0].y = self.ground_rect_pair[1].y = 500 * self.res_ratio       #y

        self.reward = 0
        self.bird = Bird(self.res, self.res_ratio)
        self.pipes = []
        self.pipes.append(Pipes(600*self.res_ratio, self.res, self.res_ratio))
        self.pipes.append(Pipes((600*self.res_ratio+(self.res[0]-90*self.res_ratio)/2+90*self.res_ratio), self.res, self.res_ratio))
        self.clock = pygame.time.Clock()
        self.pressed = 0
        self.ticks = 0
        self.animation = 0
        self.done = 0

        self.state_rects = []
        self.state_rects.extend( (pygame.Rect(10, 430, 100, 10), pygame.Rect(10, 460, 100, 10))) #pygame.Rect(10, 490, 100, 10), pygame.Rect(10, 520, 100, 10)) )
        self.action_rects = []
        self.action_rects.extend( (pygame.Rect(self.bird.rect.x+self.bird.rect.width/8, 0, 10, 0), pygame.Rect(self.bird.rect.x+self.bird.rect.width/8, 0, 10, 0)) )

        self.closest_pipe_index = 0

        self.return_image = False
        self.image_resize = 0.5

    def animate_ground(self):
        speed = 10 * self.res_ratio
        self.ground_rect_pair[0].x -= 10 * self.res_ratio
        self.ground_rect_pair[1].x -= 10 * self.res_ratio

        if self.ground_rect_pair[0].x <= -577*self.res_ratio: self.ground_rect_pair[0].x = 576*self.res_ratio
        if self.ground_rect_pair[1].x <= -577*self.res_ratio: self.ground_rect_pair[1].x = 576*self.res_ratio


    #resets the environment and returns intial state
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

        #self.animate_ground()  #needs fixing
        # self.screen.fill((212, 99, 19))
        self.screen.blit(self.background_text, self.background_rect)
        self.screen.blit(self.bird.texture, self.bird.rect, area=((int)(self.animation)%3*51*self.res_ratio, 0, 51*self.res_ratio, 36*self.res_ratio))
        self.screen.blit(self.ground_text, self.ground_rect_pair[0])
        self.screen.blit(self.ground_text, self.ground_rect_pair[1])

        for p in self.pipes:
            self.screen.blit(p.txt[0], p.rect[0], area=(0,0, p.rect[0].width, self.ground_rect_pair[0].y-p.height))
            self.screen.blit(p.txt[1], p.rect[1])

        self.animation += 0.3

        if not self.CNN_res: #only for simplified state

            for s in self.state_rects: #state variables visualization
               states = self.get_state()
               s.width = states[self.state_rects.index(s)]*100
               pygame.draw.rect(self.screen, (50, 50, 255), s)

            #predictions
            sub = 2000*(q_0 - q_1)
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
        # self.clock.tick(self.FPS)


    def step(self, action):
        self.bird.move(action)
        for p in self.pipes:
            p.move()

        if self.bird.check_collision(self.pipes[0].rect[0], self.pipes[0].rect[1], self.pipes[1].rect[0], self.pipes[1].rect[1]):
            self.done = 1

        #reward for surviving
        self.reward = 0
        state = self.get_state() #may change reward
        return [state, self.reward, self.done, " "]


    def get_state(self):
        dist_x = self.pipes[self.closest_pipe_index].rect[0].x + self.pipes[self.closest_pipe_index].rect[0].width - self.bird.rect.x
        dist_y = self.pipes[self.closest_pipe_index].rect[0].y - (self.bird.rect.y + self.bird.rect.height)
        if dist_x < 0:
            self.reward = 1
            self.closest_pipe_index += 1
            if(self.closest_pipe_index)==2: self.closest_pipe_index = 0
        else:
            self.reward = 0

        if self.return_image == True:
            image = pygame.surfarray.array3d(self.screen)
            image = np.swapaxes(image, 0, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (int(image.shape[1]*self.image_resize),int(image.shape[0]*self.image_resize)), interpolation=cv2.INTER_AREA) #resize
            image = image/255 #normalization
            return image

        #return [(self.bird.rect.y+80)/580.0, (self.bird.frame+1)/16.0, dist_x/522.0, (dist_y/1000.0 + 0.5)]
        return [dist_x/522.0, (dist_y/600.0 + 0.5)]

class Bird:
    def __init__(self, res, res_ratio):
        self.res_ratio = res_ratio
        self.texture = pygame.image.load('flappy/flappybird.png')
        self.texture = pygame.transform.scale(self.texture, (153*self.res_ratio, 36*self.res_ratio))
        self.rect = self.texture.get_rect()
        self.rect.x = res[0]/2 - self.rect.width/3
        self.frame = -1
        self.rotation = 0

    def move(self, action):
        if action == 1 and self.frame < 4:
            self.frame = 6

        if self.frame > 0:
            self.rect.y -= (int)(self.frame*self.frame*0.6*self.res_ratio)
            self.frame -= 1

        elif self.frame == 0:
            self.frame -= 1
        else:
            self.rect.y += 19 * self.res_ratio


    def check_collision(self, p1, p2, p3, p4):
        self.hitbox = pygame.Rect(self.rect.x, self.rect.y, self.res_ratio*51, self.res_ratio*36)
        cl = lambda p: self.hitbox.colliderect(p)

        if cl(p1) or cl(p2) or cl(p3) or cl(p4) or self.rect.y + self.rect.height >= self.res_ratio*500 or self.rect.y <= -self.res_ratio*50:
            return True

        else:
            return False


    def reset(self):
        self.rect.y = self.res_ratio*200
        self.frame = -1



class Pipes:
    def __init__(self, position, res, res_ratio):
        self.res_ratio = res_ratio
        self.position = position
        self.scale = (90*res_ratio, 313*self.res_ratio)
        self.txt_d = pygame.image.load('flappy/pipe.png')
        self.txt_u = pygame.image.load('flappy/piper.png')
        self.txt_d = pygame.transform.scale(self.txt_d, self.scale)
        self.txt_u = pygame.transform.scale(self.txt_u, self.scale)
        self.txt = []
        self.txt.append(self.txt_d)
        self.txt.append(self.txt_u)
        self.rect = []
        self.rect.extend([self.txt_d.get_rect(), self.txt_u.get_rect()])
        self.upper_lim = int(210*self.res_ratio)
        self.bottom_lim = int(440*self.res_ratio)
        self.height = random.randint(self.upper_lim, self.bottom_lim)
        self.gap = 140 * self.res_ratio
        self.rect[0].x = self.rect[1].x = position
        self.rect[0].y, self.rect[1].y = self.height, self.height-self.rect[1].height-self.gap
        self.res = res

    def move(self):
        self.rect[0].x -= 10 * self.res_ratio
        self.rect[1].x -= 10 * self.res_ratio
        if self.rect[0].x < -self.rect[0].width:
            self.rect[0].x = self.res[0]
            self.rect[1].x = self.res[0]
            self.height = random.randint(self.upper_lim, self.bottom_lim)
            self.rect[0].y, self.rect[1].y = self.height, self.height-self.rect[1].height-self.gap

    def reset(self):
        self.rect[0].x = self.rect[1].x = self.position
        self.height = random.randint(self.upper_lim, self.bottom_lim)
        self.rect[0].y, self.rect[1].y = self.height, self.height-self.rect[1].height-self.gap
