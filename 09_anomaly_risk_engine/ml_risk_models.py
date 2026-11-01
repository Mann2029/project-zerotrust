import math
import numpy as np

class PureIsolationTreeNode:
    """A single node in a pure Python/NumPy Isolation Tree."""
    def __init__(self, left=None, right=None, split_feature=None, split_value=None, size=1):
        self.left = left
        self.right = right
        self.split_feature = split_feature
        self.split_value = split_value
        self.size = size
        self.is_leaf = left is None and right is None

class PureIsolationTree:
    """Constructs an individual isolation tree by recursively partitioning random feature intervals."""
    def __init__(self, max_depth: int = 8):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X: np.ndarray, current_depth: int = 0) -> PureIsolationTreeNode:
        n_samples, n_features = X.shape
        if current_depth >= self.max_depth or n_samples <= 1:
            return PureIsolationTreeNode(size=n_samples)

        feat_idx = np.random.randint(0, n_features)
        min_val = X[:, feat_idx].min()
        max_val = X[:, feat_idx].max()

        if min_val == max_val:
            return PureIsolationTreeNode(size=n_samples)

        split_val = np.random.uniform(min_val, max_val)
        left_mask = X[:, feat_idx] < split_val
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return PureIsolationTreeNode(size=n_samples)

        left_child = self.fit(X[left_mask], current_depth + 1)
        right_child = self.fit(X[right_mask], current_depth + 1)

        return PureIsolationTreeNode(
            left=left_child,
            right=right_child,
            split_feature=feat_idx,
            split_value=split_val,
            size=n_samples
        )

    def path_length(self, x: np.ndarray, node: PureIsolationTreeNode, current_depth: int = 0) -> float:
        if node.is_leaf:
            return current_depth + self._c(node.size)
        
        if x[node.split_feature] < node.split_value:
            return self.path_length(x, node.left, current_depth + 1)
        else:
            return self.path_length(x, node.right, current_depth + 1)

    @staticmethod
    def _c(n: int) -> float:
        """Average path length of unsuccessful search in a Binary Search Tree (BST)."""
        if n <= 1:
            return 0.0
        if n == 2:
            return 1.0
        euler_gamma = 0.5772156649
        return 2.0 * (math.log(n - 1) + euler_gamma) - (2.0 * (n - 1) / n)


class PureNumPyIsolationForest:
    """ML #1: Unsupervised Random Tree Ensemble for Multi-Dimensional Behavioral Anomaly Detection."""
    def __init__(self, n_trees: int = 50, max_samples: int = 128):
        self.n_trees = n_trees
        self.max_samples = max_samples
        self.trees = []
        self.c_factor = 1.0

    def fit(self, X: np.ndarray):
        n_samples = len(X)
        subsample_size = min(self.max_samples, n_samples)
        max_depth = int(math.ceil(math.log2(max(subsample_size, 2))))
        self.c_factor = PureIsolationTree._c(subsample_size)
        self.trees = []

        for _ in range(self.n_trees):
            indices = np.random.choice(n_samples, size=subsample_size, replace=False)
            tree = PureIsolationTree(max_depth=max_depth)
            tree.root = tree.fit(X[indices])
            self.trees.append(tree)

    def compute_anomaly_score(self, x: np.ndarray) -> float:
        """Computes anomaly score: s(x, n) = 2^(- E(h(x)) / c(n))."""
        if not self.trees:
            return 0.0
        paths = [tree.path_length(x, tree.root) for tree in self.trees]
        avg_path = np.mean(paths)
        if self.c_factor == 0:
            return 0.0
        score = 2.0 ** (- (avg_path / self.c_factor))
        return float(np.clip(score, 0.0, 1.0))


class PureNumPyHabitPredictor:
    """ML #2: Evaluates arrival timestamp offsets and absence momentum to forecast habit decay."""
    def __init__(self):
        # Calibrated logistic boundary weights for [arrival_offset_min, past_absences, fatigue_score, day_of_week]
        self.weights = np.array([0.18, 0.45, -0.02, 0.05], dtype=np.float32)
        self.bias = -1.2

    def predict_decay_probability(self, features: np.ndarray) -> float:
        # Sigmoid activation: P(decay) = 1 / (1 + e^-(z))
        z = np.dot(features, self.weights) + self.bias
        z = np.clip(z, -50.0, 50.0)
        prob = 1.0 / (1.0 + np.exp(-z))
        return float(np.clip(prob, 0.0, 1.0))


class AnomalyRiskEngine:
    """Combines ML #1 & ML #2 into Dynamic Risk Scoring R(t)."""
    def __init__(self):
        print("[INFO] Booting ML #1 (Pure NumPy Isolation Forest) & ML #2 (Habit Decay Predictor)...")
        self.iso_forest = PureNumPyIsolationForest(n_trees=50, max_samples=128)
        self._warmup_isolation_forest()
        self.habit_model = PureNumPyHabitPredictor()
        print("[SUCCESS] Pure NumPy Risk & Anomaly Models initialized without native DLL dependencies.")

    def _warmup_isolation_forest(self):
        """Fit baseline distribution for normal campus check-in behavior."""
        np.random.seed(42)
        normal_velocity = np.random.normal(loc=2.5, scale=0.8, size=(200, 1))
        normal_liveness = np.random.uniform(low=0.90, high=1.0, size=(200, 1))
        normal_snn = np.random.uniform(low=0.80, high=1.0, size=(200, 1))
        normal_delta = np.random.normal(loc=5.0, scale=3.0, size=(200, 1))
        
        synthetic_normal = np.hstack([normal_velocity, normal_liveness, normal_snn, normal_delta])
        self.iso_forest.fit(synthetic_normal)

    def compute_behavioral_anomaly(self, velocity_sec: float, liveness_score: float, snn_conf: float, time_delta_min: float) -> float:
        """ML #1: Computes isolation outlier probability."""
        sample = np.array([velocity_sec, liveness_score, snn_conf, time_delta_min])
        return self.iso_forest.compute_anomaly_score(sample)

    def compute_habit_decay(self, arrival_offset_min: float, past_absences: int, fatigue_score: float, day_of_week: int) -> float:
        """ML #2: Evaluates probability of chronic habit decay."""
        features = np.array([arrival_offset_min, float(past_absences), fatigue_score, float(day_of_week)], dtype=np.float32)
        return self.habit_model.predict_decay_probability(features)

    def evaluate_dynamic_risk(
        self,
        liveness_score: float,
        snn_confidence: float,
        velocity_sec: float,
        arrival_offset_min: float,
        past_absences: int = 0,
        fatigue_score: float = 100.0,
        day_of_week: int = 1
    ) -> dict:
        """Calculates total Dynamic Risk Score R(t) and returns the Zero-Trust routing decision."""
        # 1. Component evaluations
        iso_anomaly = self.compute_behavioral_anomaly(velocity_sec, liveness_score, snn_confidence, arrival_offset_min)
        habit_decay = self.compute_habit_decay(arrival_offset_min, past_absences, fatigue_score, day_of_week)

        # 2. Weighted synthesis
        w_live = 35.0
        w_match = 35.0
        w_iso = 15.0
        w_habit = 15.0

        r_live = (1.0 - max(0.0, min(1.0, liveness_score))) * w_live
        r_match = (1.0 - max(0.0, min(1.0, snn_confidence))) * w_match
        r_iso = iso_anomaly * w_iso
        r_habit = habit_decay * w_habit

        total_risk_score = round(r_live + r_match + r_iso + r_habit, 2)

        # 3. Decision Boundary
        if total_risk_score < 30.0:
            decision = "AUTO_APPROVE"
            action = "Attendance verified automatically."
        elif total_risk_score < 70.0:
            decision = "FLAG_FOR_FACULTY_DUAL_VERIFICATION"
            action = "Queued to Faculty Dashboard for quick approval."
        else:
            decision = "SECURITY_BLOCK"
            action = "Access denied: High-risk transaction parameters."

        return {
            "dynamic_risk_score": total_risk_score,
            "decision": decision,
            "action": action,
            "telemetry_breakdown": {
                "liveness_penalty": round(r_live, 2),
                "snn_confidence_penalty": round(r_match, 2),
                "isolation_forest_anomaly": round(iso_anomaly, 4),
                "xgboost_habit_decay_prob": round(habit_decay, 4)
            }
        }