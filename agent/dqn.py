from traderl import agent
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from IPython.display import clear_output


class DQN(agent.DQN):
    def evolute(self, h, h_):
        pips, profits, total_pips, total_profits, total_pip, total_profit, buy, sell = self.trade(h, h_)

        acc = np.mean(pips > 0)
        total_win = np.sum(pips[pips > 0])
        total_lose = np.sum(pips[pips < 0])
        rr = total_win / abs(total_lose)
        ev = (np.mean(pips[pips > 0]) * acc + np.mean(pips[pips < 0]) * (1 - acc)) / abs(np.mean(pips[pips < 0]))

        fig = plt.figure(figsize=(10, 5), dpi=100)
        plt.subplot(1, 2, 1)
        plt.plot(total_pips)
        plt.subplot(1, 2, 2)
        plt.plot(total_profits)
        st.pyplot(fig)

        st.write(
            f"acc = {acc}, pips = {sum(pips)}\n"
            f"total_win = {total_win}, total_lose = {total_lose}\n"
            f"rr = {rr}, ev = {ev}\n"
        )

    def plot_trade(self, train=False, test=False, period=1):
        assert train or test
        h = 0
        if test:
            h = self.test_step[0]
        elif train:
            h = np.random.randint(0, int(self.train_step[-1] - 960 * period))
        h_ = h + len(self.train_step) // 12 * period
        trend = self.y[h:h_]

        pips, profits, total_pips, total_profits, total_pip, total_profit, buy, sell = self.trade(h, h_)

        fig = plt.figure(figsize=(20, 10), dpi=100)
        plt.plot(trend, color="g", alpha=1, label="close")
        plt.plot(trend, "^", markevery=buy, c="red", label='buy', alpha=0.7)
        plt.plot(trend, "v", markevery=sell, c="blue", label='sell', alpha=0.7)

        plt.legend()
        st.pyplot(fig)

        st.write(f"pip = {np.sum(pips)}"
                 f"\naccount size = {total_profit}"
                 f"\ngrowth rate = {total_profit / self.account_size}"
                 f"\naccuracy = {np.mean(np.array(pips) > 0)}")

    def plot_result(self, w, risk=0.1):
        self.model.set_weights(w)
        self.risk = risk

        fig = plt.figure(figsize=(20, 5), dpi=100)
        plt.subplot(1, 2, 1)
        plt.plot(self.test_pip)
        plt.subplot(1, 2, 2)
        plt.plot(self.test_profit)
        st.pyplot(fig)

        ################################################################################
        self.plot_trade(train=False, test=True, period=9)
        ################################################################################
        fig = plt.figure(figsize=(10, 5))
        plt.plot(self.train_loss)
        plt.plot(self.val_loss)
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        st.pyplot(fig)

        len_ = len(self.train_loss) // 2
        fig = plt.figure(figsize=(10, 5))
        plt.plot(self.train_loss[len_:])
        plt.plot(self.val_loss[len_:])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        st.pyplot(fig)
        ################################################################################
        self.evolute(self.test_step[0], self.test_step[-1])
        ################################################################################
        fig = plt.figure(figsize=(20, 5), dpi=100)
        plt.subplot(1, 2, 1)
        plt.plot(self.train_rewards)
        plt.subplot(1, 2, 2)
        plt.plot(self.test_rewards)
        st.pyplot(fig)

        st.write(f"profits = {self.max_profit}, max profits = {self.max_profits}\n"
                 f"pips = {self.max_pip}, max pip = {self.max_pips}")
        ################################################################################
        self.evolute(self.test_step[0] - len(self.train_step), self.test_step[0])

    def train(self, epoch=40, batch_size=2056):
        for _ in range(600 // epoch):
            clear_output()
            fig = plt.figure(figsize=(10, 5))
            plt.plot(self.train_loss)
            plt.plot(self.val_loss)
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            st.pyplot(fig)
            self._train(epoch, batch_size)
            self.target_model.set_weights(self.model.get_weights())
