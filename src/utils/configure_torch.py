import torch


def configure_torch(include_shape: bool = True, printoption_profile: str = "short"):
    if include_shape:
        # Hack the Tensor repr to show its shape
        # https://discuss.pytorch.org/t/tensor-repr-in-debug-should-show-shape-first/147230/5
        normal_repr = torch.Tensor.__repr__

        def fancy_repr(self):
            if self.dim() == 0:
                return normal_repr(self)
            else:
                # only for non-boring tensors we pre-fix the shape
                return f"Shape: {tuple(self.shape)}\n{normal_repr(self)}"

        torch.Tensor.__repr__ = fancy_repr  # type: ignore

    # change some settings for faster tensor.__repr__
    # actually fixes the pydev slow warnings (which happen because the past_key_values and hidden_states contain many tensors which take some time to convert to strings via repr)
    torch.set_printoptions(profile=printoption_profile)
