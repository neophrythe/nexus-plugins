# Contributing to Nexus Plugins

Thank you for your interest in contributing to the Nexus AI Framework plugins! This document provides guidelines and instructions for contributing.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Plugin Guidelines](#plugin-guidelines)
- [Testing Requirements](#testing-requirements)
- [Documentation Standards](#documentation-standards)
- [Submission Process](#submission-process)

## 📜 Code of Conduct

Please note that this project is released with a [Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Nexus AI Framework installed
- Git
- Virtual environment (recommended)

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/neophrythe/nexus-plugins.git
cd nexus-plugins

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install Nexus AI Framework
pip install nexus-ai-framework
```

## 💻 Development Process

### 1. Fork and Clone

```bash
# Fork on GitHub, then:
git clone https://github.com/YOUR_USERNAME/nexus-plugins.git
cd nexus-plugins
git remote add upstream https://github.com/neophrythe/nexus-plugins.git
```

### 2. Create Feature Branch

```bash
git checkout -b feature/your-plugin-name
```

### 3. Develop Your Plugin

Follow the plugin structure:

```
your-plugin/
├── plugin.yaml          # Required: Plugin manifest
├── __init__.py         # Required: Plugin entry point
├── config.py           # Required: Configuration schema
├── main.py             # Required: Main logic
├── requirements.txt    # Required: Dependencies
├── README.md          # Required: Documentation
├── LICENSE            # Required: License file
├── tests/             # Required: Test suite
│   ├── __init__.py
│   ├── test_plugin.py
│   └── test_config.py
├── examples/          # Recommended: Usage examples
│   ├── basic.py
│   └── advanced.py
└── assets/           # Optional: Images, data files
    └── icon.png
```

### 4. Test Your Plugin

```bash
# Run tests
pytest tests/

# Check code coverage
pytest --cov=your_plugin tests/

# Run linting
flake8 your_plugin/
black your_plugin/
```

### 5. Document Your Plugin

Create comprehensive documentation in `README.md`:

```markdown
# Plugin Name

## Description
Brief description of what your plugin does.

## Features
- Feature 1
- Feature 2

## Installation
\```bash
nexus plugin install your-plugin
\```

## Configuration
\```yaml
your_plugin:
  option1: value1
  option2: value2
\```

## Usage
\```python
from nexus import NexusCore

nexus = NexusCore()
nexus.load_plugin('your-plugin')
\```

## API Reference
Document all public functions and classes.

## Examples
Link to examples directory.

## License
Specify license.
```

## 📝 Plugin Guidelines

### Naming Conventions

- Use lowercase with hyphens: `my-awesome-plugin`
- Be descriptive but concise
- Avoid game-specific names unless it's a game plugin

### Code Standards

- Follow PEP 8
- Use type hints
- Add docstrings to all public functions
- Keep functions small and focused
- Use meaningful variable names

### Configuration Schema

```python
# config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class PluginConfig:
    """Plugin configuration."""
    enabled: bool = True
    option1: str = "default"
    option2: int = 100
    option3: Optional[float] = None
    
    def validate(self):
        """Validate configuration."""
        if self.option2 < 0:
            raise ValueError("option2 must be positive")
```

### Plugin Class Structure

```python
# main.py
from nexus.plugins import BasePlugin
import structlog

logger = structlog.get_logger()

class YourPlugin(BasePlugin):
    """Your plugin description."""
    
    def __init__(self, config):
        super().__init__(config)
        self.name = "your-plugin"
        self.version = "1.0.0"
        
    async def on_frame(self, frame):
        """Process game frame."""
        # Your logic here
        pass
        
    async def on_action(self, action):
        """Handle game action."""
        # Your logic here
        pass
```

## 🧪 Testing Requirements

### Minimum Requirements

- **Code Coverage**: Minimum 80%
- **Test Categories**:
  - Unit tests for all public functions
  - Integration tests for plugin lifecycle
  - Configuration validation tests
  - Error handling tests

### Test Template

```python
# tests/test_plugin.py
import pytest
from your_plugin import YourPlugin
from your_plugin.config import PluginConfig

class TestYourPlugin:
    
    @pytest.fixture
    def plugin(self):
        config = PluginConfig()
        return YourPlugin(config)
    
    def test_initialization(self, plugin):
        assert plugin.name == "your-plugin"
        assert plugin.version == "1.0.0"
    
    @pytest.mark.asyncio
    async def test_on_frame(self, plugin):
        frame = np.zeros((480, 640, 3))
        result = await plugin.on_frame(frame)
        assert result is not None
```

## 📚 Documentation Standards

### Required Documentation

1. **README.md**: Complete user documentation
2. **Docstrings**: All public functions and classes
3. **Comments**: Complex logic explanations
4. **Examples**: Working code examples
5. **API Reference**: Complete API documentation

### Docstring Format

```python
def process_frame(frame: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """Process a game frame for object detection.
    
    Args:
        frame: Input frame as numpy array (H, W, C)
        threshold: Detection threshold (0.0 to 1.0)
        
    Returns:
        Dictionary containing:
            - objects: List of detected objects
            - confidence: Detection confidence scores
            - time: Processing time in ms
            
    Raises:
        ValueError: If frame is invalid
        RuntimeError: If processing fails
        
    Example:
        >>> result = process_frame(frame, threshold=0.7)
        >>> print(f"Found {len(result['objects'])} objects")
    """
```

## 📤 Submission Process

### 1. Pre-submission Checklist

- [ ] Plugin follows structure guidelines
- [ ] All tests pass
- [ ] Code coverage > 80%
- [ ] Documentation complete
- [ ] Examples provided
- [ ] No hardcoded paths
- [ ] No API keys in code
- [ ] License specified

### 2. Create Pull Request

```bash
# Push your branch
git push origin feature/your-plugin-name
```

Then create a PR on GitHub with:

**Title**: `[New Plugin] Your Plugin Name`

**Description**:
```markdown
## Description
Brief description of the plugin.

## Type of Change
- [ ] New plugin
- [ ] Bug fix
- [ ] Enhancement
- [ ] Documentation

## Testing
- [ ] Tests pass locally
- [ ] Coverage > 80%
- [ ] Tested on Windows/Linux/Mac

## Checklist
- [ ] Code follows style guide
- [ ] Documentation complete
- [ ] Examples provided
- [ ] No breaking changes
```

### 3. Review Process

1. Automated checks run (tests, linting)
2. Code review by maintainers
3. Testing in various environments
4. Documentation review
5. Merge when approved

## 🎯 Plugin Categories

When submitting, choose the appropriate category:

- **Core**: Essential functionality
- **Game-Specific**: For specific games
- **AI Enhancement**: ML/AI improvements
- **Utility**: General tools
- **Vision**: Computer vision
- **Audio**: Audio processing
- **Analytics**: Data analysis
- **Network**: Networking

## 🤝 Getting Help

- **Discord**: Join our [Discord server](https://discord.gg/YOUR_INVITE)
- **Issues**: Open an [issue](https://github.com/neophrythe/nexus-plugins/issues)
- **Discussions**: Start a [discussion](https://github.com/neophrythe/nexus-plugins/discussions)

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Thank You!

Thank you for contributing to Nexus Plugins! Your efforts help make game automation and AI training accessible to everyone.

---

<p align="center">Happy Coding! 🚀</p>