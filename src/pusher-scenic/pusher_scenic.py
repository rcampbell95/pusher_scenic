import scenic
from scenic.simulators.mujoco.simulator import MujocoSimulator
from scenic.core.scenarios import Scene


if __name__ == "__main__":
    SAMPLES = 100

    right_falls = 0
    left_falls = 0

    for sample_index in range(SAMPLES):
        simulator = MujocoSimulator(xml="", actual=False, use_default_arena=False)
        scenario = scenic.scenarioFromFile("./src/pusher-scenic/pusher.scenic")

        scene, _ = scenario.generate()
        simulation = simulator.simulate(scene, maxSteps=500000)

        result = simulation.result

        #final_state = result.finalState
        #if final_state[0].x < 0.0:
        #    left_falls += 1
        #elif final_state[0].x > 0.0:
        #    right_falls += 1 

    print("Left falls: ", left_falls)
    print("Right falls: ", right_falls)
