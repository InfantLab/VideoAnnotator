"""Tests for videoannotator namespace and module imports.

Tests that the __getattr__ mechanism in src/__init__.py correctly
forwards module imports and provides proper AttributeError for invalid imports.
"""

import sys
import pytest


class TestNamespaceImports:
    """Test videoannotator.* namespace imports work correctly."""

    def test_import_videoannotator_version(self):
        """Test that version information can be imported."""
        import videoannotator

        assert hasattr(videoannotator, "__version__")
        assert isinstance(videoannotator.__version__, str)
        assert hasattr(videoannotator, "__author__")
        assert hasattr(videoannotator, "__license__")

    def test_import_api_module(self):
        """Test that videoannotator.api can be accessed."""
        import videoannotator

        # This triggers __getattr__
        api_module = videoannotator.api

        assert api_module is not None
        assert hasattr(api_module, "main")  # api.main exists

    def test_import_registry_module(self):
        """Test that videoannotator.registry can be accessed."""
        import videoannotator

        # This triggers __getattr__
        registry_module = videoannotator.registry

        assert registry_module is not None
        assert hasattr(registry_module, "pipeline_registry")

    def test_import_storage_module(self):
        """Test that videoannotator.storage can be accessed."""
        import videoannotator

        # This triggers __getattr__
        storage_module = videoannotator.storage

        assert storage_module is not None
        assert hasattr(storage_module, "base")

    def test_import_validation_module(self):
        """Test that videoannotator.validation can be accessed."""
        import videoannotator

        # This triggers __getattr__
        validation_module = videoannotator.validation

        assert validation_module is not None
        assert hasattr(validation_module, "validator")

    def test_import_worker_module(self):
        """Test that videoannotator.worker can be accessed."""
        import videoannotator

        # This triggers __getattr__
        worker_module = videoannotator.worker

        assert worker_module is not None
        assert hasattr(worker_module, "executor")

    def test_import_invalid_module_raises_attribute_error(self):
        """Test that invalid module names raise AttributeError."""
        import videoannotator

        with pytest.raises(AttributeError, match="has no attribute 'nonexistent'"):
            _ = videoannotator.nonexistent

    def test_dir_includes_public_modules(self):
        """Test that dir(videoannotator) includes public module names."""
        import videoannotator

        names = dir(videoannotator)

        # Should include version info
        assert "__version__" in names
        assert "get_version_info" in names

        # Should include public modules
        assert "api" in names
        assert "pipelines" in names
        assert "registry" in names
        assert "storage" in names


class TestDirectModuleImports:
    """Test that imports from videoannotator.* submodules work."""

    def test_import_from_api_database(self):
        """Test import from videoannotator.api.database."""
        from videoannotator.api.database import check_database_health

        assert callable(check_database_health)

    def test_import_from_registry(self):
        """Test import from videoannotator.registry."""
        from videoannotator.registry.pipeline_registry import get_registry

        assert callable(get_registry)

    def test_import_from_storage(self):
        """Test import from videoannotator.storage."""
        from videoannotator.storage.base import StorageBackend

        assert StorageBackend is not None

    def test_import_from_validation(self):
        """Test import from videoannotator.validation."""
        from videoannotator.validation.validator import ConfigValidator

        assert ConfigValidator is not None


class TestModuleCaching:
    """Test that modules are properly cached after first access."""

    def test_module_cached_in_sys_modules(self):
        """Test that accessed modules are cached."""
        import videoannotator

        # First access
        _ = videoannotator.api

        # Should now be in sys.modules
        assert hasattr(videoannotator, "api")

        # Second access should return same object
        api1 = videoannotator.api
        api2 = videoannotator.api

        assert api1 is api2


class TestPackageStructure:
    """Test package structure and metadata."""

    def test_videoannotator_is_package(self):
        """Test that videoannotator is a proper package."""
        import videoannotator

        assert hasattr(videoannotator, "__file__")
        assert hasattr(videoannotator, "__path__")

    def test_all_attribute_present(self):
        """Test that __all__ is defined."""
        import videoannotator

        assert hasattr(videoannotator, "__all__")
        assert isinstance(videoannotator.__all__, list)

    def test_version_info_accessible(self):
        """Test that version info functions are accessible."""
        from videoannotator import get_version_info, print_version_info

        assert callable(get_version_info)
        assert callable(print_version_info)

        # Test get_version_info returns expected structure
        info = get_version_info()
        assert "version" in info
        assert "python_version" in info


class TestImportExamples:
    """Test common import patterns users might use."""

    def test_import_pattern_1_from_videoannotator(self):
        """Test: from videoannotator import api"""
        # Import the package
        import videoannotator

        # Access api module
        api = videoannotator.api

        assert api is not None

    def test_import_pattern_2_direct_from_module(self):
        """Test: from videoannotator.api.database import X"""
        from videoannotator.api.database import check_database_health

        assert callable(check_database_health)

    def test_import_pattern_3_registry_function(self):
        """Test: from videoannotator.registry.pipeline_registry import get_registry"""
        from videoannotator.registry.pipeline_registry import get_registry

        registry = get_registry()
        assert registry is not None

    def test_import_pattern_4_nested_access(self):
        """Test: videoannotator.storage.base.StorageBackend"""
        import videoannotator

        # Access nested attribute
        storage_module = videoannotator.storage
        assert hasattr(storage_module, "base")

        base_module = storage_module.base
        assert hasattr(base_module, "StorageBackend")
