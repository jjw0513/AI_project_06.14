# 인공지능응용 TermProject # 32217698 정재우
32217698 정재우


이 프로젝트는 강화학습의 Value-based와 Policy-gradient 알고리즘을 사용하여 3가지 환경에서의 성능을 비교 분석합니다. 사용된 알고리즘은 DQN, SARSA, A2C이며, OpenAI Gym의 다양한 환경에서 학습되었습니다.

# RL 알고리즘 #


DQN (Deep Q-Network)

SARSA (State-Action-Reward-State-Action)

A2C (Advantage Actor-Critic)

# 환경 #

MiniGrid Memory

에이전트가 시작 위치에 있던 사물을 기억하고 동일한 객체를 선택해야 하는 환경입니다. 성공 시 보상을 받고 환경이 재시작되며, 실패 시 보상을 받지 못하고 재시작됩니다.

CartPole

막대기(cart)가 좌우로 움직이며 균형을 유지하는 목표를 가진 환경입니다.

Pendulum



# 코드 소개 #
DQN

DQN은 오프정책(off-policy) 학습 알고리즘으로, 경험 재생과 타겟 네트워크를 사용하여 학습의 안정성과 효율성을 높입니다. 이산적 행동 공간에서 매우 효과적으로 작동하며, 연속적인 행동 공간을 이산화하여 사용할 경우에도 강력한 성능을 보입니다.

#SARSA #

SARSA는 온정책(on-policy) 학습 알고리즘으로, 현재 정책을 기반으로 행동을 선택하고 학습합니다. 이산적인 행동 공간에서 효과적으로 작동하지만, 연속적인 행동 공간에서는 비효율적인 학습을 보일 수 있습니다.

# A2C #
A2C는 액터-크리틱(Actor-Critic) 방법을 사용하여 연속적인 행동 공간에서 우수한 성능을 보일 수 있습니다. 정책 그라디언트를 사용하여 정책을 직접 학습하며, 변동성이 크지만 적절한 학습률 설정을 통해 성능을 최적화할 수 있습니다.

# 참고자료 #

GitHub Repository

OpenAI Gymnasium

DQN 참고: Blog

PyTorch CartPole 예제: Tutorial

A2C 참고: Google Colab
SARSA, DQN, A2C 구현 참고: GitHub
