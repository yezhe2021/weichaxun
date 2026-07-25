from p3e_l_common import ConditionedNativeCache, condition_payload


class ReceiverNativeConditionedCache:
    def __init__(self, index_path, capacity=2):
        self.inner = ConditionedNativeCache(index_path, capacity=capacity)
        self.index = self.inner.index
        self.entries = self.inner.entries

    def __len__(self):
        return len(self.inner)

    def correct(self, index):
        return condition_payload(self.inner.load(index), "correct_question")

    def shuffled(self, index):
        return condition_payload(
            self.inner.load(index), "correct_question_hard_shuffled_evidence"
        )

