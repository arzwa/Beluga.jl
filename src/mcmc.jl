# Whale-like adaptive MCMC engine; should think about keeping it compatible with
# MCMC for the DP/finite mixture

function operator_brates!(X, chain)
    # changes λ and μ
    for e in chain.tree.order
        # use partial recomputation and other speed-up tricks
    end
end
