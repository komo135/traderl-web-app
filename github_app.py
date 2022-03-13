import streamlit as st
from traderl import data, nn
import pandas as pd
import gc

from traderl import agent
import numpy as np
import matplotlib.pyplot as plt
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


class QRDQN(agent.QRDQN, DQN):
    def evolute(self, h, h_):
        DQN.evolute(self, h, h_)

    def plot_trade(self, train=False, test=False, period=1):
        DQN.plot_trade(self, train, test, period)

    def plot_result(self, w, risk=0.1):
        DQN.plot_result(self, w, risk)

    def train(self, epoch=50, batch_size=2056):
        DQN.train(self, epoch, batch_size)


class App:
    def __init__(self):

        self.df = None
        self.agent = None
        self.model_name = ""

    def select_data(self):
        file = None

        select = st.selectbox("", ("forex", "stock", "url or path", "file upload"))
        col1, col2 = st.columns(2)
        load_file = st.button("load file")

        if select == "forex":
            symbol = col1.selectbox("", ("AUDJPY", "AUDUSD", "EURCHF", "EURGBP", "EURJPY", "EURUSD", "GBPJPY",
                                         "GBPUSD", "USDCAD", "USDCHF", "USDJPY", "XAUUSD"))
            timeframe = col2.selectbox("", ("m15", "m30", "h1", "h4", "d1"))
            if load_file:
                self.df = data.get_forex_data(symbol, timeframe)
        elif select == "stock":
            symbol = col1.text_input("", help="enter a stock symbol name")
            if load_file:
                self.df = data.get_stock_data(symbol)
        elif select == "url or path":
            file = col1.text_input("", help="enter url or local file path")
        elif select == "file upload":
            file = col1.file_uploader("", "csv")

        if load_file and file:
            st.write(file)
            self.df = pd.read_csv(file)

        if load_file:
            st.write("Data selected")

    def check_data(self):
        f"""
        # Select Data
        """
        if isinstance(self.df, pd.DataFrame):
            st.write("Data already exists")
            if st.button("change data"):
                st.warning("data and agent have been initialized")
                self.df = None
                self.agent = None

        if not isinstance(self.df, pd.DataFrame):
            self.select_data()

    def create_agent(self, agent_name, args):
        agent_dict = {"dqn": DQN, "qrdqn": QRDQN}
        self.agent = agent_dict[agent_name](**args)

    def agent_select(self):
        if not isinstance(self.df, pd.DataFrame):
            st.warning("data does not exist.\n"
                       "please select data")
            return None

        agent_name = st.selectbox("", ("dqn", "qrdqn"), help="select agent")

        """
        # select Args
        """
        col1, col2 = st.columns(2)
        network = col1.selectbox("select network", (nn.available_network))
        network_level = col2.selectbox("select network level", (f"b{i}" for i in range(8)))
        network += "_" + network_level
        self.model_name = network

        col1, col2, col3, col4 = st.columns(4)
        lr = float(col1.text_input("lr", "1e-4"))
        n = int(col2.text_input("n", "3"))
        risk = float(col3.text_input("risk", "0.01"))
        pip_scale = int(col4.text_input("pip scale", "25"))
        col1, col2 = st.columns(2)
        gamma = float(col1.text_input("gamma", "0.99"))
        use_device = col2.selectbox("use device", ("cpu", "gpu", "tpu"))
        train_spread = float(col1.text_input("train_spread", "0.2"))
        spread = int(col2.text_input("spread", "10"))

        kwargs = {"df": self.df, "model_name": network, "lr": lr, "pip_scale": pip_scale, "n": n,
                  "use_device": use_device, "gamma": gamma, "train_spread": train_spread,
                  "spread": spread, "risk": risk}

        if st.button("create agent"):
            self.create_agent(agent_name, kwargs)
            st.write("Agent created")

    def agent_train(self):
        if agent:
            if st.button("training"):
                self.agent.train()
        else:
            st.warning("agent does not exist.\n"
                       "please create agent")

    def show_result(self):
        if self.agent:
            self.agent.plot_result(self.agent.best_w)
        else:
            st.warning("agent does not exist.\n"
                       "please create agent")

    def model_save(self):
        if agent:
            save_name = st.text_input("save name", self.model_name)
            if st.button("model save"):
                self.agent.model.save(save_name)
                st.write("Model saved.")
        else:
            st.warning("agent does not exist.\n"
                       "please create agent")

    @staticmethod
    def clear_cache():
        if st.button("initialize"):
            st.experimental_memo.clear()
            del st.session_state["app"]
            gc.collect()

            m = """
                **Initialized.**
                """
            st.markdown(m)


def sidebar():
    return st.sidebar.radio("", ("Home", "select data", "create agent", "training",
                                 "show results", "save model", "initialize"))


def home():
    md = """
    # Traderl Web Application
    This web app is intuitive to [traderl](https://github.com/komo135/trade-rl).

    # How to Execute
    1. select data
        * Click on "select data" on the sidebar to choose your data.
    2. create agent
        * Click "create agent" on the sidebar and select an agent name and arguments to create an agent.
    3. training
        * Click on "training" on the sidebar to train your model.
    4. show results
        * Click "show results" on the sidebar to review the training results.
    """
    st.markdown(md)


if __name__ == "__main__":
    st.set_page_config(layout="wide", )

    if "app" in st.session_state:
        app = st.session_state["app"]
    else:
        app = App()

    select = sidebar()

    if select == "Home":
        home()

    if select == "select data":
        app.check_data()
    elif select == "create agent":
        app.agent_select()
    elif select == "training":
        app.training()
    elif select == "save model":
        app.model_save()
    elif select == "show results":
        app.show_result()

    st.session_state["app"] = app
    if select == "initialize":
        app.clear_cache()

