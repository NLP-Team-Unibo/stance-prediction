import copy

class EarlyStopping:
    def __init__(self, model, lower_is_better=False, patience=1):
        assert patience >= 0
        self.model = model
        self.patience = patience+1
        self.lower_is_better = lower_is_better
        self.best_score = -float("inf")
        self.remaining_patience = self.patience
        self.best_weights = None
    
    def __call__(self, score: float) -> bool:
        score = -score if self.lower_is_better else score
        if score > self.best_score:
            self.k = self.patience
            self.best_score = score
            self.best_weights = copy.deepcopy(self.model.state_dict())
        else:
            self.remaining_patience -= 1
            if self.remaining_patience == 0:
                return True
        return False