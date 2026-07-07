from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import LinearSVC

from sklearn.calibration import CalibratedClassifierCV

from modt.modt import MoDT

from src.weighted_qda import WeightedQDA

class HeteroMix(MoDT):

    def _update_gating(self):

        super()._update_gating()

        shift = self.gating_weights[:, -1].copy()

        self.gating_weights -= shift[:, None]

    def _train_trees(self):

        self.DT_experts = []

        experts = [

            DecisionTreeClassifier(
                max_depth=self.max_depth
            ),

            CalibratedClassifierCV(
                LinearSVC(
                    C=1.0,
                    max_iter=5000,
                    dual=False
                ),
                method="sigmoid"
            ),

            WeightedQDA(reg_param=1e-2)

        ]

        for j, model in enumerate(experts):

            model.fit(
                self.X,
                self.y,
                sample_weight=self.gating_values[:, j]
            )

            self.DT_experts.append(model)

