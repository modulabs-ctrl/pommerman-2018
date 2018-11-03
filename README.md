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

## 포머맨 플레이그라운드 클론
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
python ctrl/cli/train_pommerman.py \
--agents=tensorforce::ppo,test::agents.SimpleAgent,test::agents.SimpleAgent,test::agents.SimpleAgent \
--num_of_episodes=1 \
--max_timesteps=1000 \
--config=PommeTeamCompetition-v0 \
--render

2) 실행
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


### 남겨진 문제점들
#### 1. 터미널 실행 시에는 execute 함수의 파라메터 전달이 action 으로 호출되나, Code 에서 실행 시에는 actions 라고 호출되어 오류가 나는 이유?

