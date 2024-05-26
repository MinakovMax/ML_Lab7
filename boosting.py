from __future__ import annotations
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from scipy import sparse
import matplotlib.pyplot as plt

def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])

class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        """
        Инициализация класса Boosting.

        Параметры
        ----------
        base_model_params : dict, optional (default=None)
            Параметры базовой модели.
        n_estimators : int, optional (default=10)
            Количество базовых моделей.
        learning_rate : float, optional (default=0.1)
            Коэффициент обучения.
        subsample : float, optional (default=0.3)
            Доля данных для подвыборки.
        early_stopping_rounds : int, optional (default=None)
            Количество итераций для ранней остановки.
        plot : bool, optional (default=False)
            Флаг для построения графиков ошибок.
        """
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)
        self.scaler = StandardScaler(with_mean=False)

    def fit_new_base_model(self, X, residuals):
        """
        Обучает новую базовую модель и возвращает её вместе с оптимальным значением гаммы.

        Параметры
        ----------
        X : array-like, форма (n_samples, n_features)
            Массив признаков для набора данных.
        residuals : array-like, форма (n_samples,)
            Остатки модели.

        Возвращает
        ----------
        model : объект модели
            Обученная базовая модель.
        gamma : float
            Оптимальное значение гаммы.
        """
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, int(self.subsample * n_samples), replace=False)
        
        print(f"Number of samples: {n_samples}")
        print(f"Generated indices: {indices[:10]} (showing first 10 indices)")
        print(f"Max index: {np.max(indices)}, Min index: {np.min(indices)}")
        
        if np.max(indices) >= n_samples:
            raise ValueError(f"Index {np.max(indices)} is out of bounds for axis 0 with size {n_samples}")
        
        X_sample = X[indices]
        residuals_sample = residuals[indices]
        
        print(f"Fitting new base model with {X_sample.shape[0]} samples")
        
        model = self.base_model_class(**self.base_model_params)
        model.fit(X_sample, residuals_sample)
        
        pred = model.predict(X)
        print(f"Predictions from new base model before gamma adjustment: {pred[:5]}")
        gamma = np.sum(residuals * pred) / np.sum(pred * pred)

        print(f"New base model fitted - Gamma: {gamma}")
        print(f"Predictions from new base model after gamma adjustment: {pred[:5] * gamma}")

        return model, gamma
    
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Обучает модель на тренировочном наборе данных и выполняет валидацию на валидационном наборе.

        Параметры
        ----------
        X : array-like, форма (n_samples, n_features)
            Массив признаков для тренировочного набора.
        y : array-like, форма (n_samples,)
            Массив целевых значений для тренировочного набора.
        X_val : array-like, форма (n_samples, n_features), optional
            Массив признаков для валидационного набора.
        y_val : array-like, форма (n_samples,), optional
            Массив целевых значений для валидационного набора.
        """
        self.initial_prediction = np.mean(y)  # Initialization using mean
        print(f"Initial prediction (mean): {self.initial_prediction}")

        X = self.scaler.fit_transform(X)
        if X_val is not None:
            X_val = self.scaler.transform(X_val)
        
        y_pred = np.full(y.shape, self.initial_prediction)
        if X_val is not None:
            y_val_pred = np.full(y_val.shape, self.initial_prediction)
        
        train_errors = []
        val_errors = []
        
        for i in range(self.n_estimators):
            residuals = self.loss_derivative(y, y_pred)
            print(f"Iteration {i} - Residuals: {residuals[:5]}")
            print(f"Iteration {i} - Predictions before update: {y_pred[:5]}")

            # Ensure data consistency
            if X.shape[0] != y.shape[0]:
                raise ValueError(f"Shape mismatch: X has {X.shape[0]} samples, but y has {y.shape[0]} samples")

            new_model, gamma = self.fit_new_base_model(X, residuals)
            self.models.append(new_model)
            self.gammas.append(gamma)

            print(f"Iteration {i} - Gamma: {gamma}")

            new_predictions = self.learning_rate * gamma * new_model.predict(X)
            print(f"Iteration {i} - New model predictions: {new_predictions[:5]}")

            y_pred -= new_predictions
            print(f"Iteration {i} - Updated predictions: {y_pred[:5]}")

            train_error = self.loss_fn(y, y_pred)
            train_errors.append(train_error)
            print(f"Iteration {i} - Train Error: {train_error}")
            
            if X_val is not None:
                y_val_pred -= self.learning_rate * gamma * new_model.predict(X_val)
                val_error = self.loss_fn(y_val, y_val_pred)
                val_errors.append(val_error)
                print(f"Iteration {i} - Validation Error: {val_error}")

                if self.early_stopping_rounds and i > self.early_stopping_rounds:
                    if val_errors[-1] > min(val_errors[-self.early_stopping_rounds:]):
                        print(f"Early stopping at iteration {i}")
                        break

        if self.plot:
            plt.plot(train_errors, label='Training Error')
            if X_val is not None:
                plt.plot(val_errors, label='Validation Error')
            plt.xlabel('Iterations')
            plt.ylabel('Log Loss')
            plt.legend()
            plt.show()

    def predict_proba(self, X):
        """
        Вычисляет вероятности принадлежности классу для каждого образца.

        Параметры
        ----------
        X : array-like, форма (n_samples, n_features)
            Массив признаков для набора данных.

        Возвращает
        ----------
        probabilities : array-like, форма (n_samples, 2)
            Вероятности для каждого класса.
        """
        X = self.scaler.transform(X)
        y_pred = np.full((X.shape[0],), self.initial_prediction)
        for model, gamma in zip(self.models, self.gammas):
            y_pred -= self.learning_rate * gamma * model.predict(X)
        probabilities = self.sigmoid(y_pred)
        return np.vstack((1 - probabilities, probabilities)).T

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        """
        Находит оптимальное значение гаммы для минимизации функции потерь.

        Параметры
        ----------
        y : array-like, форма (n_samples,)
            Целевые значения.
        old_predictions : array-like, форма (n_samples,)
            Предыдущие предсказания ансамбля.
        new_predictions : array-like, форма (n_samples,)
            Новые предсказания базовой модели.

        Возвращает
        ----------
        gamma : float
            Оптимальное значение гаммы.
        """
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def score(self, x, y):
        """
        Вычисляет метрику ROC-AUC для предсказаний модели.

        Параметры
        ----------
        x : array-like, форма (n_samples, n_features)
            Массив признаков для набора данных.
        y : array-like, форма (n_samples,)
            Массив целевых значений.

        Возвращает
        ----------
        score : float
            Значение метрики ROC-AUC.
        """
        return score(self, x, y)
        
    @property
    def feature_importances_(self):
        """
        Возвращает важность признаков в обученной модели.

        Возвращает
        ----------
        importances : array-like, форма (n_features,)
            Важность каждого признака.

        Примечания
        ----------
        Важность признаков определяется по вкладу каждого признака в финальную модель.
        """
        # Инициализация массива для хранения суммарной важности признаков
        total_importances = np.zeros(self.models[0].n_features_in_)
        
        for i, model in enumerate(self.models):
            model_importances = model.feature_importances_
            print(f"Model {i} feature importances: {model_importances}")
            total_importances += model_importances
        
        print(f"Total importances before normalization: {total_importances}")
        
        # Нормализация важностей признаков
        normalized_importances = total_importances / np.sum(total_importances)
        
        print(f"Normalized importances: {normalized_importances}")
        
        return normalized_importances