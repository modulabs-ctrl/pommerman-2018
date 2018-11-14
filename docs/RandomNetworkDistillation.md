
# Random Network Distillation

[Reinforcement Learning with Prediction-Based Rewards](https://blog.openai.com/reinforcement-learning-with-prediction-based-rewards)

## Summary of Article
* Explore their environments through curiosity
* For an agent to achieve a desired goal it must first explore what is possible in its environment and what constitutes progress towards the goal.
* Simple exploration strategies are highly unlikely to gather any rewards, or see more than a few of the 24 rooms in the level.

### Why Exploration is Difficult
* many of the more complicated games require long sequences of very specific actions to experience any reward, and such sequences are extremely unlikely to occur randomly.

### Simplifying Exploration with Demonstrations
* Our approach works by letting each RL episode start from a state in a previously recorded demonstration.
* Once the agent is able to beat or at least tie the score of the demonstrator on the remaining part of the game in at least 20% of the rollouts, we slowly move the starting point back in time.
* We keep doing this until the agent is playing from the start of the game, without using the demo at all, at which point we have an RL-trained agent beating or tying the human expert on the entire game.

### Comparison to imitation-based approaches

* our method will thus not overfit to a potentially sub-optimal demonstration and could offer benefits in multi-player games where we want to optimize performance against other opponents than just the one from the demonstration.


### References
* [github of random-network-distillation](https://github.com/openai/random-network-distillation)
* [Reverse curriculum generation for robotics](https://bair.berkeley.edu/blog/2017/12/20/reverse-curriculum/)


```python
class FeatureNet(nn.Module):
    def __init__(self, obs_space):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(obs_space, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        out = self.head(x)
        obs_feature = self.fc(out).reshape(out.shape[0], -1)

        return obs_feature

def get_norm_params(obs_memory):
    global obs_apace

    obses = [[] for _ in range(obs_space)]
    for obs in obs_memory:
        for j in range(obs_space):
            obses[j].append(obs[j])

    mean = np.zeros(obs_space, np.float32)
    std = np.zeros(obs_space, np.float32)
    for i, obs_ in enumerate(obses):
        mean[i] = np.mean(obs_)
        std[i] = np.std(obs_)
    return mean, std

def calculate_reward_in(pred_net, rand_net, obs):
    global mean
    global std

    norm_obs = normalize_obs(obs, mean, std)
    state = torch.tensor([norm_obs]).to(device).float()
    with torch.no_grad():
        pred_obs = pred_net(state)
        rand_obs = rand_net(state)
        reward = (pred_obs - rand_obs).pow(2).sum()
        clipped_reward = torch.clamp(reward, -1, 1)

    return clipped_reward.item()
```


# 최초 1500번만 데이터에 대한 global mean, std 값을 설정한다.
mean, std = get_norm_params(obs_memory)

# 학습용 리워드는 초기 탐험을 장려하기 위하여 원래 리워드에 무조건 더해준다.


1. 초기 1500 timesteps 동안 데이터를 통해서 다른 에이전트들의 확률분포의 평균과 표준편차를 구한다.
obs_memory = [] # 초기 1500개 분포저장용 obs

while not is_learned:
	
	obs = env.reset()
	_ = env.step(action)

    obs_memory.append(obs)
	
	if len(obs_memory) >= 1500:
		is_learned = True
    	mean, std = get_norm_params(obs_memory)
    	obs_memory.clear()

return mean, std



2. 한 에피소드가 종료되기 까지의 모든 obs, reward, done 여부를 저장한다.

# make memory
rep_memory = deque(maxlen=memory_size)


while not is_episode_done:
	
	prev_obs = env.reset()
	next_obs, reward, done, _ = env.step(action)

    reward_in = calculate_reward_in(pred_net, rand_net, _obs)

	rep_memory.append(next_obs)
	prev_obs = next_obs

	if done:
		is_episode_done = True


learn(pred_net, rand_net, pred_optim, rep_memory)

return reward + reward_in



3. 하나의 에피소드가 끝나면 저장된 학습 데이터를 이용하여 

# make four nerual networks
obs_space = len(obs)
pred_net = FeatureNet(obs_space)
rand_net = FeatureNet(obs_space)

# make optimizer
pred_optim = torch.optim.Adam(pred_net.parameters(), lr=LR_RND, eps=EPS)



def learn(pred_net, rand_net, pred_optim, rep_memory):
    global mean
    global std

    pred_net.train()
    rand_net.train()

    train_data = []
    train_data.extend(random.sample(rep_memory, BATCH_SIZE))

    dataloader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        pin_memory=use_cuda
    )

    # training pred_net
    for i, (next_obs) in enumerate(dataloader):

        _s_norm = featurize(next_obs)
        _s_norm_batch = torch.tensor(_s_norm).to(device).float()

        pred_features = pred_net(_s_norm_batch)
        rand_features = rand_net(_s_norm_batch)

        f_loss = (pred_features - rand_features).pow(2).sum(dim=1).mean()
        pred_optim.zero_grad()
        f_loss.backward()
        pred_optim.step()


reward_in = calculate_reward_in(pred_net, rand_net, _obs)





