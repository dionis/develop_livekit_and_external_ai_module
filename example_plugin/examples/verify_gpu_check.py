
import unittest
from unittest.mock import patch, MagicMock
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Mock livekit and its submodules
sys.modules["livekit"] = MagicMock()
sys.modules["livekit.rtc"] = MagicMock()
sys.modules["livekit.agents"] = MagicMock()
sys.modules["livekit.agents.utils"] = MagicMock()
sys.modules["livekit.agents.tts"] = MagicMock()

# Determine the correct path to import from
import os
current_dir = os.getcwd()
# Assuming we run from livekit-plugins-ditto
sys.path.insert(0, current_dir)

try:
    from livekit.plugins.ditto.ditto_sdk import DittoSDKWrapper
except ImportError:
    # If running from project root
    sys.path.append(os.path.join(current_dir, "livekit-plugins-ditto"))
    from livekit.plugins.ditto.ditto_sdk import DittoSDKWrapper

class TestDittoGPUCheck(unittest.TestCase):
    
    @patch('livekit.plugins.ditto.ditto_sdk.torch.cuda.is_available')
    def test_no_gpu_raises_error(self, mock_is_available):
        # Simulate no GPU
        mock_is_available.return_value = False
        
        wrapper = DittoSDKWrapper(ditto_path="dummy_path")
        
        # Expect RuntimeError when loading
        with self.assertRaises(RuntimeError) as cm:
            wrapper.load()
        
        self.assertIn("Ditto requires a CUDA-capable GPU", str(cm.exception))
        print("\nSUCCESS: RuntimeError raised as expected when no GPU is present.")

    @patch('livekit.plugins.ditto.ditto_sdk.torch.cuda.is_available')
    @patch('livekit.plugins.ditto.ditto_sdk.sys.path') # Mock sys.path to avoid side effects
    def test_gpu_available_proceeds(self, mock_sys_path, mock_is_available):
        # Simulate GPU available
        mock_is_available.return_value = True
        
        # We also need to mock the import of stream_pipeline_online inside existing load()
        # triggering an ImportError is fine, as long as it passes the GPU check.
        # However, we want to verify it PASSED the check.
        
        wrapper = DittoSDKWrapper(ditto_path="dummy_path")
        
        try:
            wrapper.load()
        except RuntimeError as e:
            # If it raises the GPU error, fail
            if "Ditto requires a CUDA-capable GPU" in str(e):
                self.fail("Raised GPU error despite GPU being available")
            # Other errors are expected since we are mocking
            print(f"\nCaught expected downstream error: {e}")
        except Exception as e:
            print(f"\nCaught expected downstream error: {e}")
            
        print("SUCCESS: Did not raise GPU error when GPU is present.")

if __name__ == '__main__':
    unittest.main()
