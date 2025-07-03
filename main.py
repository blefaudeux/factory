import torch
from actors import (
    Feeder,
    Processor,
    Writer,
    FeederParams,
    ProcessorParams,
    WriterParams,
)
import ray


def orchestrate(max_concurrent_tasks: int):
    print(f"Orchestrating with {max_concurrent_tasks} concurrent tasks")

    # Create a processing pipeline per GPU (could be another criteria)
    n_gpus = torch.cuda.device_count()

    def create_pipeline(i):
        writer = Writer.remote(WriterParams(name=f"writer_{i}"))
        processor = Processor.remote(
            ProcessorParams(name=f"processor_{i}"), writer=writer
        )
        feeder = Feeder.remote(FeederParams(name=f"feeder_{i}"), processor=processor)
        return feeder, processor, writer

    print(f"Creating {n_gpus} pipelines - one per GPU")
    pipelines = [create_pipeline(i) for i in range(n_gpus)]

    # Start the feeders
    processing_lanes = [feeder.read.remote() for feeder, _, _ in pipelines]

    # Wait for all tasks to complete, from the feeder perspective
    _ = ray.get(processing_lanes)

    # NOTE: at this point there could be dangling tasks in the writer,
    # but the destructor will wait for them to complete
    # alternative is to do it explicitly with the following
    # _ = ray.get([processor.finish.remote() for _, processor, _ in pipelines])


if __name__ == "__main__":
    orchestrate(max_concurrent_tasks=10)
