import ray
import random
random.seed(0)


class EpisodeBuffer():
    def __init__(self):
        self.episode_buffer = []
    
    def store_episodes(self, episodes):
        self.episode_buffer.extend(episodes)
    
    def sample_batch(self, batch_size):
        return random.sample(self.episode_buffer, batch_size)


class Learner():
    def __init__(self, initial_weight):
        self.model_weight = initial_weight

    def update(self, episode_batch):
        self.model_weight = random.randint(1, 100)
    
    def get_model_weight(self):
        return self.model_weight


@ray.remote
class RolloutWorker():
    def __init__(self, initial_weight):
        self.copied_model_weight = initial_weight
    
    def generate_episode(self):
        episode = random.randint(1, 100)
        for _ in range(100000000):
            episode *= 10
            episode /= 10

        return int(episode)

    def set_weight(self, weight):
        self.copied_model_weight = weight


n_cpus = 8
n_gpus = 8
n_workers = 10
batch_size = 8
total_step = 1000

ray.init(address="auto", num_cpus=n_cpus, num_gpus=n_gpus)

workers = [RolloutWorker.remote(initial_weight=-1) for _ in range(n_workers)]
learner = Learner(initial_weight=0)
buffer = EpisodeBuffer()


for i in range(total_step):
    # Copy learner weights to workers
    copy_operations = [worker.set_weight.remote(weight=learner.get_model_weight()) for worker in workers]
    ray.get(copy_operations)

    # Get episodes
    episode_operations = [worker.generate_episode.remote() for worker in workers]
    episodes = ray.get(episode_operations)

    # Store episode & update weight
    buffer.store_episodes(episodes)
    learner.update(buffer.sample_batch(batch_size))

    print(buffer.episode_buffer)
