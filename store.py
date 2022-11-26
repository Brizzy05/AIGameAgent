# Agent registry dictionary
AGENT_REGISTRY = {}

# define decorator for registering game agents
def register_agent(agent_name=""):

    # Decorator
    def decorator(func):

        # Agent not in registry
        if agent_name not in AGENT_REGISTRY:
            AGENT_REGISTRY[agent_name] = func

        # Agent in registry
        else:

            # Assertion Error
            raise AssertionError(f"Agent {AGENT_REGISTRY[agent_name]} is already registered.")
        return func

    # Return the decorator
    return decorator
