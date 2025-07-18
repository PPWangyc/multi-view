#!/usr/bin/env python3
"""
Tests for utility functions in src.utils.utils.
"""

import os
import json
import pytest
from src.utils.utils import (
    save_training_config,
    save_training_config_summary,
    save_environment_info,
    save_all_training_info
)


class TestConfigSaving:
    """Test class for configuration saving functions."""
    
    def test_save_training_config(self, temp_dir, sample_config, sample_training_info):
        """Test save_training_config function."""
        config_path = save_training_config(sample_config, sample_training_info, temp_dir)
        
        # Verify file was created
        assert os.path.exists(config_path)
        assert config_path.endswith('.json')
        
        # Verify file content is valid JSON
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        # Check that config and training_info are in the saved data
        assert 'config' in data
        assert 'training_info' in data
        assert data['config'] == sample_config
        assert data['training_info'] == sample_training_info
    
    def test_save_training_config_summary(self, temp_dir, sample_training_info):
        """Test save_training_config_summary function."""
        summary_path = save_training_config_summary(sample_training_info, temp_dir)
        
        # Verify file was created
        assert os.path.exists(summary_path)
        assert summary_path.endswith('.txt')
        
        # Verify file content is readable
        with open(summary_path, 'r') as f:
            content = f.read()
        
        assert len(content) > 0
        # Check that key training info is present in the summary
        assert str(sample_training_info['epochs']) in content
        assert str(sample_training_info['learning_rate']) in content
        assert sample_training_info['model_name'] in content
    
    def test_save_environment_info(self, temp_dir, mock_args):
        """Test save_environment_info function."""
        env_path = save_environment_info(mock_args, temp_dir)
        
        # Verify file was created
        assert os.path.exists(env_path)
        assert env_path.endswith('.txt')
        
        # Verify file content is readable
        with open(env_path, 'r') as f:
            content = f.read()
        
        assert len(content) > 0
        # Check that command line arguments are present
        assert str(mock_args.seed) in content
        assert mock_args.config in content
    
    def test_save_all_training_info(self, temp_dir, sample_config, sample_training_info, mock_args):
        """Test save_all_training_info function."""
        saved_files = save_all_training_info(sample_config, sample_training_info, mock_args, temp_dir)
        
        # Verify all expected files were created
        expected_file_types = ['config', 'summary', 'environment']
        for file_type in expected_file_types:
            assert file_type in saved_files
            assert os.path.exists(saved_files[file_type])
    
    def test_saved_files_are_readable(self, temp_dir, sample_config, sample_training_info, mock_args):
        """Test that all saved files are readable and contain valid content."""
        saved_files = save_all_training_info(sample_config, sample_training_info, mock_args, temp_dir)
        
        for file_type, file_path in saved_files.items():
            assert os.path.exists(file_path), f"File {file_type} not found at {file_path}"
            
            # Try to read the file
            if file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                assert isinstance(data, dict), f"JSON file {file_type} should contain a dictionary"
            elif file_path.endswith('.txt'):
                with open(file_path, 'r') as f:
                    content = f.read()
                assert len(content) > 0, f"Text file {file_type} should not be empty"
    
    def test_file_paths_are_unique(self, temp_dir, sample_config, sample_training_info, mock_args):
        """Test that all saved files have unique paths."""
        saved_files = save_all_training_info(sample_config, sample_training_info, mock_args, temp_dir)
        
        file_paths = list(saved_files.values())
        unique_paths = set(file_paths)
        
        assert len(file_paths) == len(unique_paths), "All saved files should have unique paths"
    
    def test_files_are_in_temp_directory(self, temp_dir, sample_config, sample_training_info, mock_args):
        """Test that all files are saved in the specified temporary directory."""
        saved_files = save_all_training_info(sample_config, sample_training_info, mock_args, temp_dir)
        
        for file_type, file_path in saved_files.items():
            assert file_path.startswith(temp_dir), f"File {file_type} should be in temp directory" 