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
