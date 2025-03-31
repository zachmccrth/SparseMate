import torch as t
from torch.utils.data import DataLoader

from model_tools.truncated_leela import TruncatedModel
import torch.multiprocessing as mp
import queue




def _background_buffer_filler(
        cfg,  # dictionary with config info
        data_q: mp.Queue,
        done_event: mp.Event,
        device: t.device = t.device("cpu"),
):
    """
    Runs in a background process.
    Continuously fills a backup buffer with fresh activations
    and pushes them onto the data_q for the main process to consume.
    """

    layer = cfg['layer']
    submodule = TruncatedModel(onnx_model_path=cfg['onnx_model_path'],layer=int(layer) ,device=device)
    submodule.eval()  # Ensure it's in evaluation mode
    dataset_class = cfg['dataset_class']
    dataset = dataset_class()
    dataloader = DataLoader(dataset, batch_size=None, batch_sampler=None)
    data_generator = iter(dataloader)
    while not done_event.is_set():
        try:
            # Create a fresh backup buffer on CPU:
            backup_buffer = t.empty(
                cfg['SIZE_OF_BUFFER_IN_TOKENS'],
                cfg['d_submodule'],
                dtype=cfg['dtype'],
                device=device,
            )

            idx = 0
            while idx < cfg['SIZE_OF_BUFFER_IN_TOKENS']:
                # 1) get next data from your generator
                #    In a real scenario, you might need to handle StopIteration or break out.
                boards = []
                for _ in range(cfg['refresh_batch_size_boards']):
                    boards.append(next(data_generator))
                inputs = submodule.make_inputs(boards)
                with t.no_grad():
                    hidden_states = submodule(inputs)

                # Fill the backup_buffer
                remaining_space = cfg['SIZE_OF_BUFFER_IN_TOKENS'] - idx
                hidden_states = hidden_states[:remaining_space]
                backup_buffer[idx: idx + len(hidden_states)] = hidden_states
                idx += len(hidden_states)


            # 3) Once the backup buffer is filled, put it in the queue
            data_q.put(backup_buffer)

        except StopIteration:
            # If your data generator is exhausted, you might signal the main process
            print("Background filler: data exhausted.")
            done_event.set()
            break

    print("Background filler process: done_event is set. Exiting.")
class LeelaImpActivationBuffer:
    def __init__(self,
                 dataset_class,
                 onnx_model_path,
                 layer,
                 d_submodule=768,
                 io='out',
                 n_ctxs=4000,
                 refresh_batch_size=50,
                 out_batch_size=10,
                 device=t.device("cpu"),
                 dtype=t.float32,
                 ):

        if io not in ['in', 'out']:
            raise ValueError("io must be either 'in' or 'out'")

        if d_submodule is None:
            raise ValueError("d_submodule cannot be inferred and must be specified directly")

        self.device = device
        self.dtype = dtype

        self.size_of_buffer_in_boards = n_ctxs
        self.BOARD_SIZE = 64
        self.SIZE_OF_BUFFER_IN_TOKENS = n_ctxs * self.BOARD_SIZE
        self.refresh_batch_size_boards = refresh_batch_size
        self.OUT_BATCH_SIZE_TOKENS = out_batch_size
        self.d_submodule = d_submodule

        # The main, actively used activation buffer is on GPU (or whatever self.device is)
        self.activation_buffer = t.empty(
            self.SIZE_OF_BUFFER_IN_TOKENS, d_submodule,
            device=self.device, dtype=self.dtype
        )
        self.backup_buffer = t.empty(
            self.SIZE_OF_BUFFER_IN_TOKENS, d_submodule,
            device=device, dtype=self.dtype
        )

        self.onnx_model_path = onnx_model_path

        self.current_token_idx = len(self.activation_buffer)

        self.dataset_class = dataset_class

        # We'll create a Multiprocessing Queue
        self.data_q = mp.Queue(maxsize=2)  # keep small, e.g. 2, so we don't fill up memory
        # We'll also use an Event to signal the worker to stop
        self.done_event = mp.Event()

        # Build a config dict to pass to the worker
        self.cfg = dict(
            SIZE_OF_BUFFER_IN_TOKENS=self.SIZE_OF_BUFFER_IN_TOKENS,
            d_submodule=self.d_submodule,
            dtype=self.dtype,
            dataset_class=self.dataset_class,
            BOARD_SIZE=self.BOARD_SIZE,
            refresh_batch_size_boards=self.refresh_batch_size_boards,
            onnx_model_path=onnx_model_path,
            layer=layer,
        )

        # Start the worker process
        self.process = mp.Process(
            target=_background_buffer_filler,
            args=(self.cfg, self.data_q, self.done_event, self.device),
        )
        self.process.daemon = True
        self.process.start()

    def __iter__(self):
        return self

    def __next__(self):
        """
        Return a batch of activations from the current activation_buffer.
        If near the end, swap in a new buffer from the queue.
        """
        with t.no_grad():
            # If we are close to exhausting the current GPU buffer,
            # swap in the next buffer from the queue (if available).
            if self.current_token_idx + self.OUT_BATCH_SIZE_TOKENS >= self.SIZE_OF_BUFFER_IN_TOKENS:
                self.swap_in()

            # Get the slice:
            idxs = self.read_order[self.current_token_idx: self.current_token_idx + self.OUT_BATCH_SIZE_TOKENS]
            self.current_token_idx += len(idxs)
            return self.activation_buffer[idxs]

    def swap_in(self):
        """
        Attempt to retrieve a new buffer from the queue, move it to self.device,
        and make it the new activation_buffer.
        """
        # Mark old activation_buffer for GC or reuse.
        # Wait for the worker to produce the backup buffer:
        try:
            new_backup_buffer = self.data_q.get(timeout=60)  # wait up to 30s
            # Move it to device (GPU or otherwise) for immediate usage
            new_backup_buffer = new_backup_buffer.to(self.device)
            # Swap:
            self.activation_buffer = new_backup_buffer
            # Re-init read order:
            self.read_order = t.randperm(len(self.activation_buffer), device=self.device)
            self.current_token_idx = 0
        except queue.Empty:
            print("Warning: Timed out waiting for backup buffer from worker!")
            # Optionally raise StopIteration or handle more gracefully.
            raise StopIteration("No more data in queue.")


    def close(self):
        """
        Signal the worker to stop and clean up.
        """
        self.done_event.set()
        self.process.join(timeout=10)
        self.data_q.close()
        self.data_q.join_thread()
        # If your input_data_generator needs cleanup:
        # self.input_data_generator.close()


    @property
    def config(self):
        return {
            'd_submodule': self.d_submodule,
            'n_ctxs': self.size_of_buffer_in_boards,
            'ctx_len': self.BOARD_SIZE,
            'refresh_batch_size': self.refresh_batch_size_boards,
            'out_batch_size': self.OUT_BATCH_SIZE_TOKENS,
            'device': str(self.device),
            'onnx_model_path': self.onnx_model_path
        }

