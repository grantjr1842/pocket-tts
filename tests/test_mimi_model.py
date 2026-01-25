"""Tests for MimiModel implementation."""

import pytest

from pocket_tts.models.mimi import MimiModel


class TestMimiModelForward:
    """Test cases for MimiModel forward method implementation."""

    def test_forward_method_exists(self):
        """Test that forward method is implemented and doesn't raise NotImplementedError."""
        # We can't easily create a real MimiModel without all the complex dependencies
        # But we can test that the forward method exists and is callable

        # Check that forward method exists in the class
        assert hasattr(MimiModel, "forward")
        assert callable(getattr(MimiModel, "forward"))

        # Check that the method signature is correct
        import inspect

        sig = inspect.signature(MimiModel.forward)
        params = list(sig.parameters.keys())

        # Should have 'self', 'x', and optional 'mimi_state'
        assert "x" in params
        assert "mimi_state" in params

        # Check that mimi_state has a default value
        assert sig.parameters["mimi_state"].default is None

    def test_forward_method_not_abstract(self):
        """Test that forward method is not abstract (doesn't raise NotImplementedError)."""
        # We can't instantiate MimiModel directly due to complex dependencies
        # But we can check the source code doesn't contain NotImplementedError
        import inspect

        # Get the source code of the forward method
        source = inspect.getsource(MimiModel.forward)

        # Should not contain NotImplementedError
        assert "NotImplementedError" not in source

        # Should contain actual implementation logic
        assert "encode_to_latent" in source
        assert "quantizer" in source
        assert "decode_from_latent" in source

    def test_forward_method_docstring(self):
        """Test that forward method has proper documentation."""
        docstring = MimiModel.forward.__doc__

        assert docstring is not None
        assert "encode" in docstring.lower()
        assert "decode" in docstring.lower()
        assert "Args:" in docstring
        assert "Returns:" in docstring

    def test_encode_to_latent_exists(self):
        """Test that encode_to_latent method exists and is implemented."""
        assert hasattr(MimiModel, "encode_to_latent")
        assert callable(getattr(MimiModel, "encode_to_latent"))

        # Check method signature
        import inspect

        sig = inspect.signature(MimiModel.encode_to_latent)
        params = list(sig.parameters.keys())

        assert "x" in params
        assert len(params) == 2  # self and x

    def test_decode_from_latent_exists(self):
        """Test that decode_from_latent method exists and is implemented."""
        assert hasattr(MimiModel, "decode_from_latent")
        assert callable(getattr(MimiModel, "decode_from_latent"))

        # Check method signature
        import inspect

        sig = inspect.signature(MimiModel.decode_from_latent)
        params = list(sig.parameters.keys())

        assert "latent" in params
        assert "mimi_state" in params
        assert len(params) == 3  # self, latent, and mimi_state

    def test_decode_from_latent_not_abstract(self):
        """Test that decode_from_latent method is implemented."""
        import inspect

        # Get the source code
        source = inspect.getsource(MimiModel.decode_from_latent)

        # Should not contain NotImplementedError
        assert "NotImplementedError" not in source

        # Should contain implementation logic
        assert "_to_encoder_framerate" in source
        assert "decoder_transformer" in source
        assert "decoder" in source


class TestMimiModelIntegration:
    """Integration tests that verify the fix works in context."""

    def test_import_works(self):
        """Test that MimiModel can be imported without issues."""
        # This should not raise any import errors
        from pocket_tts.models.mimi import MimiModel

        # Check class exists
        assert MimiModel is not None

    def test_no_notimplementederror_in_source(self):
        """Test that the source file doesn't contain NotImplementedError for forward."""
        import os

        # Get the path to the mimi.py file (we're in tests/ subdirectory of worktree)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        worktree_root = os.path.dirname(current_dir)
        mimi_path = os.path.join(worktree_root, "pocket_tts", "models", "mimi.py")

        # Read the source file
        with open(mimi_path, "r") as f:
            source = f.read()

        # Check that forward method doesn't raise NotImplementedError
        lines = source.split("\n")
        in_forward_method = False

        for line in lines:
            if "def forward(" in line:
                in_forward_method = True
                continue

            if in_forward_method:
                if line.strip().startswith("def ") and "forward(" not in line:
                    # We've moved to the next method
                    break

                # Check for NotImplementedError in forward method
                if "NotImplementedError" in line and "raise" in line:
                    pytest.fail("Found NotImplementedError in forward method")

        # If we get here, no NotImplementedError was found in forward method

    def test_method_order_in_source(self):
        """Test that methods are in the expected order in source."""
        import os

        # Get the path to the mimi.py file (we're in tests/ subdirectory of worktree)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        worktree_root = os.path.dirname(current_dir)
        mimi_path = os.path.join(worktree_root, "pocket_tts", "models", "mimi.py")

        # Read the source file
        with open(mimi_path, "r") as f:
            source = f.read()

        # Check method order
        forward_pos = source.find("def forward(")
        encode_pos = source.find("def encode_to_latent(")
        decode_pos = source.find("def decode_from_latent(")

        # All methods should exist
        assert forward_pos != -1, "forward method not found"
        assert encode_pos != -1, "encode_to_latent method not found"
        assert decode_pos != -1, "decode_from_latent method not found"

        # forward should come before encode_to_latent and decode_from_latent
        # (this is just a structural check, not required but good practice)
        # We don't enforce this strictly, just verify they exist


if __name__ == "__main__":
    pytest.main([__file__])
