from random import randint

max_num = 50
balls_no = 10

arr = []
is_picked = [0 for i in range(max_num)]

for c in range(balls_no):
    new_idx = randint(0, balls_no - c - 1)

    idx = 0