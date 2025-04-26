import models


class MembershipFunctionFactory:
    @staticmethod
    def create_membership_function(mf_type: str, params: list[float] | None = None) -> models.MembershipFunction:
        if mf_type not in models.MembershipFunctionType:
            raise ValueError(f"Invalid membership function type: {mf_type}")

        if mf_type == models.MembershipFunctionType.GAUSSIAN:
            return models.GaussianMembershipFunction(params=params)
        if mf_type == models.MembershipFunctionType.GAUSSIAN2:
            return models.Gaussian2MembershipFunction(params=params)
        if mf_type == models.MembershipFunctionType.TRAPEZOID:
            return models.TrapezoidMembershipFunction(params=params)
        if mf_type == models.MembershipFunctionType.TRIANGLE:
            return models.TriangleMembershipFunction(params=params)
        if mf_type == models.MembershipFunctionType.SIGMOID:
            return models.SigmoidMembershipFunction(params=params)

        return models.BellMembershipFunction(params=params)
