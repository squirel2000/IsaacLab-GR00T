import torch
import numpy as np

class LowPassFilter:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.prev_output = None  # Store the last smoothed output from the previous batch

    def filter(self, actions):
        """
        Apply a first-order low-pass filter to the action sequence
        actions: torch.Tensor, shape [16, 1, 28]
        Returns: smoothed_actions (shape [16, 1, 28]), last_output (shape [1, 28])
        """
        smoothed_actions = torch.zeros_like(actions)
        smoothed_actions[0] = actions[0]  # Initialize the first frame
        if self.prev_output is not None:
            smoothed_actions[0] = self.alpha * actions[0] + (1 - self.alpha) * self.prev_output
        
        for t in range(1, actions.shape[0]):
            smoothed_actions[t] = self.alpha * actions[t] + (1 - self.alpha) * smoothed_actions[t-1]
        
        self.prev_output = smoothed_actions[-1]  # Update the last frame
        return smoothed_actions


class MovingAverageFilter:
    def __init__(self, window_size=3):
        """
        Initialize the moving average filter
        window_size: Size of the averaging window (integer, recommended 3 or 5)
        """
        self.window_size = window_size
        self.prev_output = None  # Store the last smoothed output from the previous batch
        self.buffer = []  # Store previous frames for cross-batch continuity

    def filter(self, actions):
        """
        Apply a moving average filter to the action sequence
        actions: torch.Tensor, shape [16, 1, 28]
        Returns: smoothed_actions (shape [16, 1, 28]), last_output (shape [1, 28])
        """
        smoothed_actions = torch.zeros_like(actions)
        extended_actions = actions.clone()

        # If there is previous batch data, add it to the buffer for cross-batch continuity
        if self.prev_output is not None:
            self.buffer.append(self.prev_output)
            if len(self.buffer) > self.window_size - 1:
                self.buffer.pop(0)
            # Prepend buffer data to the current batch
            extended_actions = torch.cat([torch.stack(self.buffer, dim=0), actions], dim=0)

        # Apply moving average
        for t in range(actions.shape[0]):
            start = max(0, t + len(self.buffer) - self.window_size + 1)
            end = t + len(self.buffer) + 1
            smoothed_actions[t] = torch.mean(extended_actions[start:end], dim=0)

        self.prev_output = smoothed_actions[-1]  # Update the last frame
        self.buffer.append(self.prev_output)  # Update the buffer
        if len(self.buffer) > self.window_size - 1:
            self.buffer.pop(0)

        return smoothed_actions