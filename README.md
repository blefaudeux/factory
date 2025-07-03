A simple example of how to use the Ray actors abstraction to create a set of three-steps pipeline (read-process-write).
This can be used to process data on GPU for instance. There are _way_ more sophisticated ways to do this, but they often come with hidden costs (materialization of the data, etc..), so this simple take can be useful at times.

Bottlenecks:
- multiple cuda streams handled in a thread pool to better saturate the GPUs. Easy to implement with this structure, typically brings ~15% speedup.
- ray will typically start all the actors in a process, which is nice but means that you can occur IPC costs (payload serialization, ..). This will eventually bottleneck the speed of this approach