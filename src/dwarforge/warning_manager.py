from threading import local

# Thread-local storage for warnings
warning_storage = local()


def set_warnings(warnings):
    warning_storage.warnings = warnings


def get_warnings():
    return getattr(warning_storage, 'warnings', [])


def clear_warnings():
    if hasattr(warning_storage, 'warnings'):
        del warning_storage.warnings
