import ray
from dataclasses import dataclass
import torch
from typing import List


@dataclass
class Params:
    name: str
    max_tasks_in_flight: int = 5


@dataclass
class FeederParams(Params):
    max_samples: int = 50


@dataclass
class WriterParams(Params):
    pass


@dataclass
class ProcessorParams(Params):
    pass


@dataclass
class Data:
    id: int


class ActorsRequestManager:
    def __init__(self, max_tasks_in_flight: int):
        print(
            f"Initializing RequestManager with {max_tasks_in_flight} concurrent tasks"
        )
        self.max_tasks_in_flight = max_tasks_in_flight
        self.futures: List[ray.ObjectRef] = []

    def handle_task(self, callback):
        # Make sure that not too many tasks are already in flight
        if len(self.futures) >= self.max_tasks_in_flight:
            ready, self.futures = ray.wait(self.futures, num_returns=1)
            _ = ray.get(ready[0])  # you can do something with the returned values

        # Add the task to the list of futures
        self.futures.append(callback())

    def finish(self):
        ready, self.futures = ray.wait(self.futures, num_returns=len(self.futures))
        print("All tasks completed")
        _ = ray.get(ready)  # you can do something with the returned values


@ray.remote
class Writer:
    def __init__(self, writer_params: WriterParams) -> None:
        # do something to init the writer
        print(f"Initializing {writer_params.name}")

    def write(self, data) -> None:
        # dummy..
        print("Writing data")
        pass


@ray.remote(  # type: ignore
    num_gpus=1,
    max_concurrency=10,
)  # Processor is GPU enabled + define actual max concurrency
class Processor:
    def __init__(self, processor_params: ProcessorParams, writer: Writer) -> None:
        # do something to init the processor
        print(f"Initializing {processor_params.name}")

        # grab a handle to the next step in the pipeline
        self.writer = writer

        # use a request manager to make sure that not too many tasks are in flight
        self.request_manager = ActorsRequestManager(
            max_tasks_in_flight=processor_params.max_tasks_in_flight
        )

        # run all the GPU tasks in a threadpool, with a cuda stream per thread
        # this makes sure that we can saturate GPU use even if tehre's some CPU-GPU communication
        self.max_tasks_in_flight = processor_params.max_tasks_in_flight
        self.cuda_streams = [
            torch.cuda.Stream() for _ in range(processor_params.max_tasks_in_flight)
        ]
        self.current_cuda_stream = 0

    def process(self, data: Data) -> bool:
        self.current_cuda_stream = (self.current_cuda_stream + 1) % len(
            self.cuda_streams
        )

        with torch.cuda.stream(self.cuda_streams[self.current_cuda_stream]):
            print(
                f"Starting processing {data.id} on stream {self.current_cuda_stream} "
            )

            # do something to process the data
            # we'll do something dummy here which does CPU->GPU and does some GPU computations
            # this is dumb but gets to show that the threadpool and cuda streams make sure that the GPU
            # is saturated even if there's some CPU-GPU communication
            dummy_inputs = torch.rand(10000, 10000, device="cpu")
            dummy_inputs = dummy_inputs.to("cuda", non_blocking=True)
            dummy_outputs = torch.cos(
                torch.sqrt(torch.matmul(dummy_inputs, dummy_inputs))
            )
            dummy_outputs = dummy_outputs.cpu()
            print(f"Processing finished for {data.id}")

            # when done, send some data to the writer
            # this is all dummy, we don't even send what we computed here, this is just an example
            self.request_manager.handle_task(lambda: self.writer.write.remote(data))  # type: ignore
            return True  # Could be something else

    def finish(self):
        # Wait for all the delayed actors communications to finish
        self.request_manager.finish()

    def __del__(self):
        self.finish()


@ray.remote
class Feeder:
    def __init__(self, feeder_params: FeederParams, processor: Processor) -> None:
        # do something to init the feeder
        print(f"Initializing {feeder_params.name}")
        self.max_samples = feeder_params.max_samples
        self.name = feeder_params.name

        # grab a handle to the next step in the pipeline
        self.processor = processor

        # use a request manager to make sure that not too many tasks are in flight
        self.request_manager = ActorsRequestManager(
            max_tasks_in_flight=feeder_params.max_tasks_in_flight
        )

    def read(self) -> None:
        # Dummy, replace with actual data reading
        for i in range(self.max_samples):
            self.request_manager.handle_task(
                lambda: self.processor.process.remote(Data(id=i))  # type: ignore
            )

        # Could be different, but we enforce here that the feeder will not return before all of its requests have returned
        self.finish()

    def finish(self):
        self.request_manager.finish()

    def __del__(self):
        self.finish()
