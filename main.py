from agents import Agent
from environment import Environment

def main():
    env = Environment()
    agent = Agent()

    while not env.done:
        action = agent.act(env.state)
        env.step(action)

if __name__ == "__main__":
    main()
