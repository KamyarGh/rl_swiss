'''
Make the meta replay buffer have this behaviour:
sample a task
if task in buffer, empty it and replace with K rollouts
else
    if num tasks in buffer == N:
        delete the least recently updated task buffer
        add K rollouts for the sampled task
'''
