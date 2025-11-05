#!/bin/bash

# Setup Claude CLI
# This script installs the Anthropic Python SDK and sets up the Claude CLI

echo "Setting up Claude CLI..."

# Install Anthropic SDK
pip3 install anthropic

# Create ~/.local/bin if it doesn't exist
mkdir -p ~/.local/bin

# Check if claude script exists
if [ ! -f ~/.local/bin/claude ]; then
    echo "Creating Claude CLI wrapper..."
    cat > ~/.local/bin/claude << 'EOF'
#!/usr/bin/env python3
"""
Simple Claude CLI wrapper using Anthropic API
Usage: claude "your question here"
"""

import sys
import os
from anthropic import Anthropic

def main():
    # Check for API key
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("\nTo set it up:")
        print("1. Get your API key from: https://console.anthropic.com/")
        print("2. Add to ~/.bashrc:")
        print("   export ANTHROPIC_API_KEY='your-api-key-here'")
        print("3. Run: source ~/.bashrc")
        sys.exit(1)
    
    # Get user input
    if len(sys.argv) < 2:
        print("Usage: claude 'your question here'")
        print("Example: claude 'What is the capital of France?'")
        sys.exit(1)
    
    user_message = ' '.join(sys.argv[1:])
    
    # Initialize client
    client = Anthropic(api_key=api_key)
    
    # Send message
    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": user_message
            }]
        )
        
        # Print response
        print(message.content[0].text)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF
    chmod +x ~/.local/bin/claude
fi

# Add to PATH if not already there
if ! grep -q "/home/nvidia/.local/bin" ~/.bashrc 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    echo "Added ~/.local/bin to PATH in ~/.bashrc"
fi

echo ""
echo "✓ Claude CLI setup complete!"
echo ""
echo "Next steps:"
echo "1. Get your API key from: https://console.anthropic.com/"
echo "2. Add to ~/.bashrc:"
echo "   export ANTHROPIC_API_KEY='your-api-key-here'"
echo "3. Run: source ~/.bashrc"
echo "4. Test: claude 'Hello, Claude!'"
echo ""

