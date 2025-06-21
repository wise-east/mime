MODEL2CLASS = {}

try:
    from .internvl2_5 import InternVL2_5
    MODEL2CLASS["internvl2_5"] = InternVL2_5
except ImportError as e:
    print(f"Could not import InternVL2_5: {e}. Please check installation instructions.")

try:
    from .janus_model import JanusVLM
    MODEL2CLASS["janus"] = JanusVLM
except ImportError as e:
    print(f"Could not import JanusVLM: {e}. Please check https://github.com/deepseek-ai/Janus for installation instructions.")

try:
    from .openai_model import OpenAIVLM
    MODEL2CLASS["openai"] = OpenAIVLM
except ImportError as e:
    print(f"Could not import OpenAIVLM: {e}. Please check installation instructions.")

try:
    from .qwen2vl import Qwen2VL
    MODEL2CLASS["qwen2vl"] = Qwen2VL
except ImportError as e:
    print(f"Could not import Qwen2VL: {e}. Please check installation instructions.")

try:
    from .qwen25vl import Qwen25VL
    MODEL2CLASS["qwen25vl"] = Qwen25VL
except ImportError as e:
    print(f"Could not import Qwen25VL: {e}. Please check installation instructions.")

try:
    from .gemini import GeminiVLM
    MODEL2CLASS["gemini"] = GeminiVLM
except ImportError as e:
    print(f"Could not import GeminiVLM: {e}. Please check installation instructions.")

try:
    from .phi35 import PhiVLM
    MODEL2CLASS["phi35"] = PhiVLM
except ImportError as e:
    print(f"Could not import PhiVLM: {e}. Please check installation instructions.")
