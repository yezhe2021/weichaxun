from p3e_b_common import SenderNativeHeadwiseCache
from p3e_l_common import ConditionedNativeCache, condition_payload


class FreshReaderMemory:
    def __init__(self, base_index, conditioned_index):
        self.base = SenderNativeHeadwiseCache(base_index, capacity=4)
        self.conditioned = ConditionedNativeCache(conditioned_index, capacity=2)
        if len(self.base) != len(self.conditioned):
            raise RuntimeError("Evidence-only and Q-conditioned cache sizes differ")

    def __len__(self):
        return len(self.base)

    def evidence_only(self, index):
        return self.base.load(index)

    def question_conditioned(self, index):
        return condition_payload(self.conditioned.load(index), "correct_question")

    def hard_index(self, index):
        return int(self.conditioned.entries[index]["hard_evidence_index"])

    def evidence_only_hard(self, index):
        return self.base.load(self.hard_index(index))

    def question_conditioned_hard(self, index):
        return condition_payload(
            self.conditioned.load(index),
            "correct_question_hard_shuffled_evidence",
        )
