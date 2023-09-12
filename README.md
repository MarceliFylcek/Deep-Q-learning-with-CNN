# Deep Q-learning

DQN (based on Mnih et al., 2015) on Flappy Bird

## To test the game

```bash
python game.py
```
Up-arrow key to jump.

## To train with x, y coordinates
```bash
python main_xy_state.py
```

## To train with game frames
```
python main_CNN.py
```


## X, Y state (no velocity):
Relative distance from the bird to the nearest obstacle

![image](https://github.com/MarceliFylcek/Deep-Q-learning-with-CNN/assets/101202474/681e9165-12d3-499a-8019-e159b5a29c32)

### Normalized value map dependent on position

![image](https://github.com/MarceliFylcek/Deep-Q-learning-with-CNN/assets/101202474/dd5a87e2-b6c9-4120-afed-0868fd2cc53e)


## Game frames state:
Last 3 game frames

![image](https://github.com/MarceliFylcek/Deep-Q-learning-with-CNN/assets/101202474/000aad4a-019f-41a7-9ad2-3549ea3900e6)

## Results


The length of the red bar represents the expected discounted sum of future rewards for falling down. When the length gets to zero the agent jumps. Blue bars represent x and y variables.

https://user-images.githubusercontent.com/101202474/181769894-f90a63aa-923c-49d9-aae7-015be283c1ca.mp4


