from federated_inference.common.environment import Member, DataDistributionType

class MemberConfiguration(): 
    def __init__(self, idx: int = 0, member_type: Member = Member.SERVER, input_size: tuple|None=None):
        self.idx = idx
        self.type = member_type
        self.input_size = input_size

class HFLMemberConfiguration(MemberConfiguration):
    def __init__(self, idx: int = 0, member_type: Member = Member.SERVER, input_size: tuple|None=None):
        super().__init__(idx, member_type, input_size)
        if member_type == Member.SERVER:
            self.N_ROUNDS = 40
            self.CLIENTS_PER_ROUND = 5