from traderl import agent
from agent.dqn import DQN


class QRDQN(agent.QRDQN, DQN):
    def evolute(self, h, h_):
        DQN.evolute(self, h, h_)

    def plot_trade(self, train=False, test=False, period=1):
        DQN.plot_trade(self, train, test, period)

    def plot_result(self, w, risk=0.1):
        DQN.plot_result(self, w, risk)

    def train(self, epoch=50, batch_size=2056):
        DQN.train(self, epoch, batch_size)