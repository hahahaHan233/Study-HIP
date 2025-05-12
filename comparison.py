import random
from pyhip import HIP
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pickle


class Comparison:
    def __init__(self, data_source, video_id, num_train, num_test):
        """
        data_source: dict[str, (daily_share, daily_view, daily_watch)]
        """
        self.video_id = video_id
        self.daily_share = data_source[video_id][0]
        self.daily_view = data_source[video_id][1]
        self.num_train = num_train
        self.num_test = num_test

        self.x = self.daily_share
        self.x_train = self.daily_share[:num_train]
        self.x_test = self.daily_share[num_train:num_train + num_test]

        self.y = self.daily_view
        self.y_train = self.daily_view[:num_train]
        self.y_test = self.daily_view[num_train:num_train + num_test]

        self.num_initialization = 25
        self.order = (2, 1, 2)

    def arimax_predict(self):
        # from pmdarima import auto_arima
        #
        # model = auto_arima(
        #     y=self.y_train,
        #     exogenous=self.x_train,
        #     seasonal=False,
        #     stepwise=True,
        #     suppress_warnings=True,
        #     error_action="ignore"
        # )
        #
        # forecast = model.predict(n_periods=len(self.y_test), exogenous=self.x_test)

        model = ARIMA(endog=self.y_train, exog=self.x_train, order=self.order)
        model_fit = model.fit(method_kwargs={"maxiter": 500})
        forecast = model_fit.forecast(steps=len(self.y_test), exog=self.x_test)

        return forecast

    def hip_predict(self):
        hip_model = HIP()
        hip_model.initial(self.daily_share, self.daily_view, self.num_train, self.num_test, self.num_initialization)
        hip_model.fit_with_bfgs()
        view_pred = hip_model.predict(params=hip_model.get_parameters_abbr(), x=hip_model.x)
        forecast = view_pred[self.num_train:self.num_train + self.num_test]
        return forecast

    def evaluate_rmse(self, y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def test_initialization_effect(self, init_list=[1,5,10,20,25]):
        rmse_list = []

        for init in init_list:
            try:
                hip_model = HIP()
                hip_model.initial(self.daily_share, self.daily_view,
                                  self.num_train, self.num_test, num_initialization=init)
                hip_model.fit_with_bfgs()
                pred = hip_model.predict(params=hip_model.get_parameters_abbr(), x=hip_model.x)
                pred_test = pred[self.num_train:self.num_train + self.num_test]
                rmse = np.sqrt(mean_squared_error(self.y_test, pred_test))
                rmse_list.append(rmse)
                print(f"[init={init}] HIP RMSE: {rmse:.4f}")
            except Exception as e:
                print(f"[init={init}] HIP failed: {e}")
                rmse_list.append(np.nan)

        # 可视化 RMSE 随初始化窗口的变化
        plt.figure(figsize=(8, 5))
        plt.plot(init_list, rmse_list, marker='o', linestyle='-', color='teal')
        plt.title(f"Effect of num_initialization on HIP RMSE (Video {self.video_id})")
        plt.xlabel("num_initialization")
        plt.ylabel("HIP RMSE")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return dict(zip(init_list, rmse_list))

    def plot_predictions(self, pred_arimax, pred_hip, rmse_arimax,rmse_hip,title="Forecast Comparison"):
        y_true = self.y_test
        arimax_pred = pred_arimax
        hip_pred = pred_hip
        days = np.arange(len(y_true))

        plt.figure(figsize=(10, 5))
        plt.plot(days, y_true, label="True Views", color="black", linewidth=2)
        plt.plot(days, arimax_pred, label=f"ARIMAX (RMSE={rmse_arimax:.2f})", linestyle="--")
        plt.plot(days, hip_pred, label=f"HIP (RMSE={rmse_hip:.2f})", linestyle=":")
        plt.xlabel("Test Day")
        plt.ylabel("Daily Views")
        plt.title(title or f"Forecast Comparison for VideoID = {self.video_id}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_series(self, title="Original Time Series with Train/Test Split"):
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        age = self.num_train + self.num_test
        ax1.plot(np.arange(age), self.y[:age], 'g-', label='observed #views')
        ax2.plot(np.arange(age), self.x[:age], 'r-', label='#share')
        ax1.plot((self.num_train, self.num_train), (ax1.get_ylim()[0], ax1.get_ylim()[1]), 'k--')

        ax1.set_xlim(xmin=0)
        ax1.set_xlim(xmax=age)
        ax1.set_ylim(ymin=max(0, ax1.get_ylim()[0]))
        ax2.set_ylim(ymin=max(0, ax2.get_ylim()[0]))
        ax2.set_ylim(ymax=3 * max(self.x))
        ax1.set_xlabel('video age (day)')
        ax1.set_ylabel('Number of views', color='g')
        ax1.tick_params('y', colors='g')
        ax2.set_ylabel('Number of shares', color='r')
        ax2.tick_params('y', colors='r')

        ax1.set_title(title)


        plt.legend([plt.Line2D((0, 1), (0, 0), color='g'),
                    plt.Line2D((0, 1), (0, 0), color='r')],
                   ['Observed view', 'Observed share'],
                   frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.125),
                   fancybox=True, shadow=True, ncol=4)

        plt.tight_layout(rect=[0, 0.04, 1, 1])
        plt.show()

def batch_evaluate_models(data_source, n_sample=10, num_train=90, num_test=30, verbose=True):
    results = {}
    selected_ids = random.sample(list(data_source.keys()), n_sample)

    arimax_rmses = []
    hip_rmses = []

    for vid in selected_ids:
        try:
            cmp = Comparison(data_source, video_id=vid, num_train=num_train, num_test=num_test)

            pred_arimax = cmp.arimax_predict()
            pred_hip = cmp.hip_predict()

            y_test = cmp.y_test

            arimax_rmse = cmp.evaluate_rmse(y_test, pred_arimax)
            hip_rmse = cmp.evaluate_rmse(y_test, pred_hip)
            arimax_rmses.append(arimax_rmse)
            hip_rmses.append(hip_rmse)
            results[vid] = {
                'arimax_rmse': arimax_rmse,
                'hip_rmse': hip_rmse
            }

            if verbose:
                print(f'-------------{vid}-------------')
                print(f"[ARIMAX] RMSE: {arimax_rmse:.4f}")
                print(f"[HIP]    RMSE: {hip_rmse:.4f}")
        except Exception as e:
            print(f"Error processing video {vid}: {e}")

    results['avg_arimax_rmse'] = np.mean(arimax_rmses)
    results['avg_hip_rmse'] = np.mean(hip_rmses)

    if verbose:
        print("\n=========== Summary ===========")
        print(f"Avg ARIMAX RMSE: {results['avg_arimax_rmse']:.4f}")
        print(f"Avg HIP    RMSE: {results['avg_hip_rmse']:.4f}")

    return results


def case_study(data_source, video_id, num_train=90, num_test=30, plot_predictions=False):
    print(f"========== Case Study for Video: {video_id} ==========")

    # Initialize
    cmp = Comparison(data_source=data_source, video_id=video_id, num_train=num_train, num_test=num_test)

    # Plot original series
    cmp.plot_series(title=f" VideoId = {video_id}")

    # Plot prediction
    if plot_predictions:
        pred_arimax = cmp.arimax_predict()
        pred_hip = cmp.hip_predict()

        rmse_arimax = cmp.evaluate_rmse(cmp.y_test, pred_arimax)
        rmse_hip = cmp.evaluate_rmse(cmp.y_test, pred_hip)

        cmp.plot_predictions(pred_arimax, pred_hip,rmse_arimax,rmse_hip, title=f"Forecast Comparison for Video {video_id}")

        print(f"[ARIMAX] RMSE: {rmse_arimax:.4f}")
        print(f"[HIP   ] RMSE: {rmse_hip:.4f}")


if __name__ == '__main__':

    active_videos = pickle.load(open('./data/active-dataset.p', 'rb'))
    num_train = 90
    num_test = 30

    n_sample = 100

    batch_evaluate_models(active_videos,  n_sample=n_sample, num_train = num_train,  num_test = num_test, verbose=True)
    # case_study(data_source=active_videos, video_id='X0ZEt_GZfkA', num_train=num_train, num_test=num_test)




