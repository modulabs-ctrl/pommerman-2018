# pommerman-2018

## 아나콘다 설치
* [아나콘다 다운로드](https://www.anaconda.com/download)
* macOS High Sierra (10.13.6)
* Ananconda 1.6.2

## 파이썬 설치
* [파이썬 다운로드](https://www.python.org/downloads/)
* Python 3.6.x

## 가상환경 구성
```bash
conda env list
conda create -n py36 python=3.6 anaconda
```

## 로컬 환경 구성
* 깃 로컬 레포지토리의 홈은 아래와 같습니다.
```bash
mkdir $HOME/git
cd $HOME/git
```

## 포머맨 플레이그라운드 설치 (소스 코드가 필요 없는 경우)
```bash
pip install git+https://github.com/modulabs-ctrl/playground.git
```

## 포머맨 플레이그라운드 클론 (소스 코드가 필요한 경우)
```bash
git clone https://github.com/MultiAgentLearning/playground.git
git clone https://github.com/modulabs-ctrl/pommerman-2018.git
```

## 가상환경 활성화 및 설치
```bash
source activate py36

cd $HOME/git/playground
python setup.py install

cd $HOME/git/pommerman-2018
pip install -r requirements.txt
```

## 학습 및 테스트
```bash
1) 학습
export episode=500
python ctrl/cli/train_pommerman.py \
--agents=tensorforce::ppo,test::agents.SimpleAgent,test::agents.SimpleAgent,test::agents.SimpleAgent \
--num_of_episodes=$episode \
--max_timesteps=1000 \
--config=PommeTeamCompetition-v0 \
--render

2) 텐서보드를 통한 디버깅
tensorboard --logdir=./temp

3) 실행
python ctrl/cli/simple_ffa_run.py



```

## 실험 히스토리
### 2018-11-02 (금) ~ 2018-11-03 (토)
#### 1. 초기 설정을 어떠한 행동(움직임+폭탄)을 하기라도 하면 +2점을 주었더니 초반에는 학습이 되는 듯 하다가, 후반에 가면 폭탄을 놓지 않는 습관을 가짐
> 행동을 하지 않으면 -1, 하면 +1, 폭탄은 +3 으로 주기로 함.
#### 2. 그랬더니 구석에서 움직여지지 않는 방향에서 대기하면서 버팀
* 폭탄의 보상이 3점 밖에 안되므로 움직이며 버텨도 3스텝이면 보상 받고도 남기 때문으로 보입
> 행동에 대한 보상을 각각 -0.2, +0.1로 변경하고 폭탄을 던지면 10점을 주기로 함.
#### 3. 초기 학습이 중요한 건지 1000 steps 를 추가 학습으로도 폭탄을 놓지 않음
#### 4. 구석에서 이동되지 않는 방향으로 이동하는 척 행동을 취하는 척 함 
> 행동이 아니라 좌표를 기준으로 이동되지 않으면 마이너스 보상을 주기로 함.
#### 5. 중간에 대기하거나, 잠시 이동하는 과정에도 마이너스 보상이 안 좋은 영향을 주는 것 같음
> 2개의 스텝을 연속으로 좌표가 움직이지 않는 경우 -0.2, 버퍼는 0, 움직이면 0.1점 보상을 주기로 함.

### 2018-11-04 (일)
#### 1. 약 1만 에피소드를 수행해 보았으나, 아주 잘 피하는데 폭탄을 던져서 승리하는 경우가 없음. 
* 보상 : --rewards="100.0, -30.0, 0.0, -0.1, 0.3, 0.2, 0.5, 0.5, 0.5"
> 다양한 파라메터로 더 다른 행태의 에이전트를 학습할 것인가?
> 폭탄을 던지게 다시 학습하게 할 것인가?


## 액션아이템
### 디버깅을 위한 텐서보드 적용
### 하이퍼 파라메터를 정하기 위한 방법론 찾아보기
### PPO 이해, 정리 및 공유
### 해외 멀티에이전트 학습 사례 방법론 리뷰 (openAI 도타)


### 남겨진 문제점들
#### 1. 터미널 실행 시에는 execute 함수의 파라메터 전달이 action 으로 호출되나, Code 에서 실행 시에는 actions 라고 호출되어 오류가 나는 이유?

### 2018-11-10 (토)
#### 1. 어떤 행동이 좋은 행동인가?
> 행동 반경이 넓고, 아이템을 먹으면서, 폭탄을 많이 쏘면서, 잘 죽지 않는 에이전트
> 죽었더라도 나의 행동이 바람직 하다면 긍정적인 보상을 받아야 마땅하다.
> 승리하더라도 폭탄을 한 번도 쓰지 않았다면 보상의 감쇄해야 하지 않을까?

### 2018-11-11 (일)
#### 1. 알파고 강의를 들으면서 느낀 점
> 잘 하는 상대화 학습을 하기에는 시간이 오히려 더 오래 걸리거나 잘 못 학습될 가능성이 있을 것 같다.
> 차라리 못 하는 애들끼리 학습을 계속 시켜서 잘 하는 애를 찍어서 성장시키는 방법을 해볼 수 있을 것 같다.
> 네트워크를 충분히 깊고 넓게 하여 학습하는 것도 고려해 보아야 겠다
> 결국 셀프 플레이 강화학습 만으로도 개선이 가능할 것 같고, 바둑의 규칙만을 적용하는 것이 핵심
> 처음부터 MCTS 와 강화학습을 응용해보는 것이 어떨까?
> 바둑의 기보와 비슷하게 보드 전체의 상태를 입력이고, 내가 선택할 수 있는 액션의 범위가 있다
> 최적화 관점에서 보드 변환의 8방향 대칭 등의 적용이 좋을 듯 하다

