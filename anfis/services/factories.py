import models


class MembershipFunctionFactory:
    @staticmethod
    def create_membership_function(mf_type: str) -> models.MembershipFunction:
        if mf_type not in models.MembershipFunctionType:
            raise ValueError(f"Invalid membership function type: {mf_type}")

        if mf_type == models.MembershipFunctionType.GAUSSIAN:
            return models.GaussianMembershipFunction()
        if mf_type == models.MembershipFunctionType.GAUSSIAN2:
            return models.Gaussian2MembershipFunction()
        if mf_type == models.MembershipFunctionType.TRAPEZOID:
            return models.TrapezoidMembershipFunction()
        if mf_type == models.MembershipFunctionType.TRIANGLE:
            return models.TriangleMembershipFunction()
        if mf_type == models.MembershipFunctionType.SIGMOID:
            return models.SigmoidMembershipFunction()

        return models.BellMembershipFunction()


class FuzzificationFactory:
    @staticmethod
    def create_fuzzifier(mf_type: str) -> models.Fuzzification:
        if mf_type not in models.MembershipFunctionType:
            raise ValueError(f"Invalid membership function type: {mf_type}")

        if mf_type == models.MembershipFunctionType.GAUSSIAN:
            return models.GaussianFuzzification()
        if mf_type == models.MembershipFunctionType.GAUSSIAN2:
            return models.Gaussian2Fuzzification()
        if mf_type == models.MembershipFunctionType.TRAPEZOID:
            return models.TrapezoidFuzzification()
        if mf_type == models.MembershipFunctionType.TRIANGLE:
            return models.TriangleFuzzification()
        if mf_type == models.MembershipFunctionType.SIGMOID:
            return models.SigmoidFuzzification()

        return models.BellFuzzification()
