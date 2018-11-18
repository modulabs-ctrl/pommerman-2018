
# Reinforcement Learning with Prediction-Based Rewards

[Reinforcement Learning with Prediction-Based Rewards](https://blog.openai.com/reinforcement-learning-with-prediction-based-rewards)

## Summary of Article
* Random Network Distillation (RND), a prediction-based method for encouraging reinforcement learning agents to explore their environments through curiosity
* RND incentivizes visiting unfamiliar states by measuring how hard it is to predict the output of a fixed random neural network on visited states.

## Progress in Montezuma’s Revenge
* For an agent to achieve a desired goal it must first explore what is possible in its environment and what constitutes progress towards the goal.
* Simple exploration strategies are highly unlikely to gather any rewards, or see more than a few of the 24 rooms in the level.

## [Large-Scale Study of Curiosity-Driven Learning](https://pathak22.github.io/large-scale-curiosity/)
* Prior to developing RND, we investigated learning *without any environment-specific rewards.*
* Curiosity gives us an easier way to teach agents to interact with any environment, rather than via an extensively engineered task-specific reward function that we hope corresponds to solving a task.
* There, the agent learns a next-state predictor model from its experience, and uses the error of the prediction as an intrinsic reward.

## What Do Curious Agents Do?
* To our surprise, in some environments the agent achieved the game’s objective even though the game’s objective was not communicated to it through an extrinsic reward.

## The Noisy-TV Problem
* the agent sometimes gets trapped by its curiosity as the result of the noisy-TV problem.
* The agent finds a source of randomness in the environment and keeps observing it, always experiencing a high intrinsic reward for such transitions.

## Random Network Distillation
* 이후-상태 예측은 noisy-TV 문제에 대해서 본질적으로 민감할 수 밖에 없기 때문에, 아래와 같은 예측 오류들을 확인할 수 있었습니다.
* 사실 1: 난이도가 높은 상황을 일반화 하지 못해 제대로 학습하지 못한 경우
* 사실 2: 학습해야 할 상황이 deterministic 하지 않아 학습이 어렵다
* 사실 3: 예측에 필요한 정보가 부족하거나, 너무 복잡한 경우라 학습이 어렵다
> 1번의 경우 극복해야 할 문제이므로 패스하고, 2~3번의 개선을 위해, 주어진 다음 상태에 대해 다음 상태를 위한 고정된 랜덤 초기화 뉴럴넷에 근거한 새로운 탐험인 RND 방법을 연구했습니다.


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





