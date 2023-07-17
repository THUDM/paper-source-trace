class Args(object):
    def __init__(self, model_type):
        if model_type == "SVM":
            self.C = 1
            self.kernel = "sigmoid"
            self.verbose = 2
            self.max_iter = 100
            self.tol = 0.1
        elif model_type == "RandomForest":
            self.n_estimators = 100
        elif model_type == "LR":
            self.solver = 'saga'
            self.multi_class = 'ovr'
