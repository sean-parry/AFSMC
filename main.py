import SMC, utils

def main():
    smc_af = utils.methods.SMC_GP(
        func_class = utils.test_functions.Branin,
        smc_obj=SMC.smc_search.SMC(
            target_obj=SMC.target_functions.gp_fit(),
            n_samples = 30,
            n_iters = 20,
            initial_proposal_obj=SMC.initial_proposals.Gauss(),
            proposal_obj=SMC.proposals.Defensive_Sampling(),
    ),
    n_iters=30)

    print(smc_af.get_regret())

if __name__ == '__main__':
    main()