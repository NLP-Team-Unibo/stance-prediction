import copy

class EarlyStopping:
    def __init__(self, model, lower_is_better=False, patience=1):
        """
            Implements the early stopping regularization.

            Parameters
            ----------
            model: nn.Module
                A reference to the model under observation.
            lower_is_better: bool
                If 'true', the current score will be considered improved if lower than the previous one, if 'false' otherwise.
            patience: int
                The maximum number of epochs between the last best score and the new one.
        """
        assert patience >= 0
        self.model = model
        self.patience = patience+1
        self.lower_is_better = lower_is_better
        self.best_score = -float("inf")
        self.remaining_patience = self.patience
        self.best_weights = None
    
    def __call__(self, score: float) -> bool:
        """
            Updates the remaining patience based on the current score and returns True when the patience has run out
            and training should stop. 
    
            Parameters
            ----------
            score: float
                Current value for the tracked metric
            
            Returns
            -------
            outcome: bool
                True if the patience has run out, False otherwise
        """
        score = -score if self.lower_is_better else score
        outcome = False
        if score > self.best_score:
            self.remaining_patience = self.patience
            self.best_score = score
            self.best_weights = copy.deepcopy(self.model.state_dict())
        else:
            self.remaining_patience -= 1
            if self.remaining_patience == 0:
                outcome = True
        return outcome