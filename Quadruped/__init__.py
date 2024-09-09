from gymnasium.envs.registration import register

register(
    id="SimpleQuad-v0.1",
    entry_point="Quadruped.Envs:QuadEnv"
)

register(
    id="SimpleQuadvecENV-v0.1",
    entry_point="Quadruped.Envs.quad_vec_env:QuadVecEnv",
    order_enforce=False
)