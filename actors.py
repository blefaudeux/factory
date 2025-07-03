import ray
from dataclasses import dataclass
import time
import random


@dataclass
class Params:
    name: str
    max_tasks_in_flight: int = 5


@dataclass
class FeederParams(Params):
    max_samples: int = 20


@dataclass
class WriterParams(Params):
    pass


@dataclass
class ProcessorParams(Params):
    pass


@dataclass
class Data:
    id: int


class RequestManager:
    def __init__(self, max_tasks_in_flight: int):
        print(
            f"Initializing RequestManager with {max_tasks_in_flight} concurrent tasks"
        )
        self.max_tasks_in_flight = max_tasks_in_flight
        self.futures = []

    def handle_task(self, callback):
        # Make sure that not too many tasks are already in flight
        if len(self.futures) >= self.max_tasks_in_flight:
            ready, self.futures = ray.wait(self.futures, num_returns=1)
            _ = ray.get(ready[0])  # you can do something with the returned values

        # Add the task to the list of futures
        self.futures.append(callback())

    def finish(self):
        ready, self.futures = ray.wait(self.futures)
        print("All tasks completed")
        _ = ray.get(ready)  # you can do something with the returned values


@ray.remote
class Writer:
    def __init__(self, writer_params: WriterParams) -> None:
        # do something to init the writer
        print(f"Initializing {writer_params.name}")

    def write(self, data) -> None:
        # dummy..
        print(f"Writing {data}")


@ray.remote(
    num_gpus=1, max_concurrency=10
)  # Processor is GPU enabled + define actual max concurrency
class Processor:
    def __init__(self, processor_params: ProcessorParams, writer: Writer) -> None:
        # do something to init the processor
        print(f"Initializing {processor_params.name}")

        # grab a handle to the next step in the pipeline
        self.writer = writer

        # use a request manager to make sure that not too many tasks are in flight
        self.request_manager = RequestManager(
            max_tasks_in_flight=processor_params.max_tasks_in_flight
        )

    def process(self, data: Data) -> None:
        # do something to process the data
        sleep_time = random.random() * 10
        print(f"Processing {data} - sleeping for {sleep_time:.2f}s", flush=True)
        time.sleep(sleep_time)

        # when done, send the data to the writer
        self.request_manager.handle_task(lambda: self.writer.write.remote(data))

    def finish(self):
        self.request_manager.finish()

    def __del__(self):
        self.finish()


@ray.remote(max_concurrency=10)  # This will define the actual concurrent execution
class Feeder:
    def __init__(self, feeder_params: FeederParams, processor: Processor) -> None:
        # do something to init the feeder
        print(f"Initializing {feeder_params.name}")
        self.max_samples = feeder_params.max_samples
        self.name = feeder_params.name

        # grab a handle to the next step in the pipeline
        self.processor = processor

        # use a request manager to make sure that not too many tasks are in flight
        self.request_manager = RequestManager(
            max_tasks_in_flight=feeder_params.max_tasks_in_flight
        )

    def read(self) -> None:
        # Dummy, replace with actual data reading
        for i in range(self.max_samples):
            self.request_manager.handle_task(
                lambda: self.processor.process.remote(Data(id=i))
            )

        self.finish()

    def finish(self):
        self.request_manager.finish()

    def __del__(self):
        self.finish()
